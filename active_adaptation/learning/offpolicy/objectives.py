from typing import Any

import torch
import torch.nn as nn
import einops
from tensordict import TensorDictBase
from torchrl.objectives import hold_out_net
from jaxtyping import Float
from active_adaptation.learning.ppo.common import ACTION_KEY, OBS_KEY


class MultiStepReturn(nn.Module):
    def __init__(self, gamma: float, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.register_buffer("gamma", torch.tensor(gamma))
        self.gamma: torch.Tensor

    def forward(
        self,
        next_observations: Float[torch.Tensor, "T N obs_dim"],
        actions: Float[torch.Tensor, "T N act_dim"],
        rewards: Float[torch.Tensor, "T N 1"],
        terminated: Float[torch.Tensor, "T N 1"],
        done: Float[torch.Tensor, "T N 1"],
    ) -> tuple[
        Float[torch.Tensor, "N obs_dim"],
        Float[torch.Tensor, "N 1"],
        Float[torch.Tensor, "N 1"],
    ]:
        T, N = next_observations.shape[:2]
        assert T == self.n_steps

        device = rewards.device
        gammas = self.gamma ** torch.arange(self.n_steps, device=device)

        cum_not_done = (~done).cumprod(dim=0)
        cum_reward = (rewards * gammas.reshape(self.n_steps, 1, 1)).cumsum(dim=0)
        alive_steps = cum_not_done.sum(dim=0)

        last_indices = alive_steps.clamp_max(self.n_steps - 1).reshape(N)
        batch_indices = torch.arange(N, device=device)

        next_observations = next_observations[last_indices, batch_indices]
        rewards = cum_reward[last_indices, batch_indices]
        terminated = terminated[last_indices, batch_indices]

        discount = (
            self.gamma
            * gammas[last_indices].reshape_as(terminated)
            * (1.0 - terminated.float())
        )

        return next_observations, rewards, discount


def _policy_dist(actor: nn.Module, obs: torch.Tensor) -> torch.distributions.Distribution:
    """Actor forward returns a ``Distribution`` (legacy code may return ``(dist, ...)``)."""
    out = actor(obs)
    return out[0] if isinstance(out, tuple) else out


def _actor_q_per_sample(critic: nn.Module, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Scalar Q per batch row for actor objectives (twin mean, or expected Q if distributional)."""
    with hold_out_net(critic):
        qs = critic(obs, act)
        if hasattr(critic, "expected_values"):
            q = critic.expected_values(qs).mean(dim=-1)
        else:
            q = qs.mean(dim=-1)
    return q.reshape(obs.shape[0])


class SACLoss:
    """Vanilla SAC actor loss on Q only: minimize ``-E[Q(s,a)]`` for ``a = rsample()`` from ``actor``. Entropy / ``alpha`` is added outside.

    Returns per-batch ``q_term`` (``-Q``), ``entropy_est`` = ``-log pi(a|s)``, the policy ``dist``, and the reparameterized ``act``.
    """

    def compute(
        self, tensordict: TensorDictBase, actor: nn.Module, critic: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        obs = tensordict[OBS_KEY]
        dist = _policy_dist(actor, obs)
        act = dist.rsample()
        q = _actor_q_per_sample(critic, obs, act)
        entropy_est = -dist.log_prob(act)
        return -q, entropy_est, dist, act


class RatioLoss:
    """Advantage-weighted log-likelihood: ``-(Q(s,a') - Q(s,a_b))_stopgrad * log pi(a'|s)`` with ``a' \sim \pi(\cdot|s)`` and ``a_b`` from the buffer."""

    def compute(
        self, tensordict: TensorDictBase, actor: nn.Module, critic: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = tensordict[OBS_KEY]
        act_buf = tensordict[ACTION_KEY]
        dist = _policy_dist(actor, obs)
        act_new = dist.rsample()
        logp = dist.log_prob(act_new)
        assert logp.shape == act_new.shape[:-1]
        logp = logp.reshape(obs.shape[0])
        q_buf = _actor_q_per_sample(critic, obs, act_buf)
        q_new = _actor_q_per_sample(critic, obs, act_new)
        adv = (q_new - q_buf).detach()
        entropy_est = -dist.log_prob(act_new)
        return -(adv * logp), entropy_est


class WeightedRegression:
    """Draw ``num_candidates`` actions per state from ``actor``; ``w_k = softmax(Q(s,a_k)/temp)``; loss ``-sum_k w_k log pi(a_k|s)``.

    Returns per-batch loss and ``entropy_est`` = mean over candidates of ``-log pi(a_k|s)`` (MC NLL, same notion as :class:`SACLoss`).
    """

    def __init__(self, num_candidates: int = 8, temperature: float = 1.0):
        if num_candidates < 2:
            raise ValueError("num_candidates must be >= 2 for softmax weights.")
        self.num_candidates = num_candidates
        self.temperature = temperature

    def compute(
        self, tensordict: TensorDictBase, actor: nn.Module, critic: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = tensordict[OBS_KEY]
        dist = _policy_dist(actor, obs)
        k = self.num_candidates
        cand_act = dist.rsample((k,))  # [K, N, act_dim]
        logp = dist.log_prob(cand_act)
        assert logp.shape == cand_act.shape[:-1]
        logp = einops.rearrange(logp, "sample batch -> batch sample")
        act_flat = einops.rearrange(cand_act, "sample batch act -> (batch sample) act")
        obs_flat = einops.repeat(obs, "batch obs -> (batch sample) obs")
        q_flat = _actor_q_per_sample(critic, obs_flat, act_flat)
        q = einops.rearrange(q_flat, "(batch sample) -> batch sample")
        w = torch.softmax(q / self.temperature, dim=-1)
        loss = -(w * logp).sum(dim=-1)
        entropy_est = (-logp).mean(dim=-1)
        return loss, entropy_est

