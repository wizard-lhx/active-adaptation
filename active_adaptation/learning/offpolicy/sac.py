import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
)

from torchrl.data import Composite, TensorSpec
from torchrl.modules.distributions import TruncatedNormal

from active_adaptation.learning.modules import ResidualMLP, MLP, VecNorm
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    GAE,
    OBS_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)

from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.distributional import (
    ValueDistribution,
    expected_q_from_logits,
)
from active_adaptation.learning.offpolicy.objectives import (
    AdvantageWeightedRegression,
    MultiStepReturn,
    SACLoss,
)
from active_adaptation.learning.offpolicy.distribution import ScaledTanhNormal


cs = ConfigStore.instance()


def gaussian_target_entropy(act_dim: int, sigma: float) -> float:
    """Differential entropy of independent \\mathcal N(0, \\sigma^2) in \\mathbb R^d (FlashSAC-style).

    H = (d/2) * log(2 * pi * e * sigma^2). Used as SAC log-alpha target when ``target_entropy_sigma`` is set.
    """
    if sigma <= 0:
        raise ValueError("target_entropy_sigma must be positive for principled entropy.")
    return 0.5 * float(act_dim) * math.log(2.0 * math.pi * math.e * sigma * sigma)


def _init_sac_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


@dataclass
class SACConfig:
    _target_: str = "active_adaptation.learning.offpolicy.sac.SAC"
    name: str = "sac"
    train_every: int = 4
    buffer_size: int = 5000
    warm_up_steps: int = 200
    lr: float = 5e-4
    # TD learning
    n_steps: int = 3
    gamma: float = 0.99
    utd_ratio: int = 4
    # architecture
    actor_init: str = "zeros"
    actor_vecnorm: Any = "pre"
    critic_vecnorm: Any = "pre"
    distributional: bool = True
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    target_action_noise: float = 0.01
    # actor objective: ``sac`` = :class:`SACLoss`; ``advantage_weighted_regression`` = :class:`AdvantageWeightedRegression` (linear A weighting, no softmax).
    actor_loss: str = "sac"
    wr_num_candidates: int = 8
    wr_temperature: float = 0.05  # AWR beta in exp(A / beta)
    wr_normalize_advantage: bool = False
    # BC-style anchor on replay actions; curbs Q exploitation (SAC and AWR).
    actor_behavior_coef: float = 0.5
    # sac specific
    entropy_bonus: float = 0.0
    # If set: H_target = (d/2)*log(2*pi*e*sigma^2) for N(0,sigma^2)^d (FlashSAC).
    # If None: use -dim(A) (common heuristic for tanh-squashed SAC).
    target_entropy_sigma: float | None = 0.15

    tau_actor: float = 0.1 # a relatively large value for faster convergence
    tau_Q: float = 0.2  # a relatively large value for faster convergence
    lr_alpha: float = 1e-2
    max_grad_norm: float = 1.0
    v_update_every: int = 32
    v_trace_steps: int = 32  # on-policy GAE horizon from replay ring (like blade_runner last())
    v_inner: int = 2
    gae_lambda: float = 0.95

    debug: bool = False
    vecnorm: bool = True

    in_keys: Tuple[str, ...] = (OBS_KEY, ACTION_KEY)


cs.store(name="sac", node=SACConfig, group="algo")


class TwinQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        activation: type[nn.Module] = nn.SiLU,
        layer_norm = "pre"
    ):
        super().__init__()
        critic_input_dim = obs_dim + act_dim
        self.critic_1 = nn.Sequential(
            ResidualMLP([critic_input_dim, 512, 512, 512], activation),
            nn.Linear(512, 1),
        )
        self.critic_2 = nn.Sequential(
            ResidualMLP([critic_input_dim, 512, 512, 512], activation),
            nn.Linear(512, 1),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.critic_1.apply(_init_sac_linear)
        self.critic_2.apply(_init_sac_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.critic_1(x)
        q2 = self.critic_2(x)
        return torch.cat([q1, q2], dim=-1)


class TwinDistributionalQNetwork(nn.Module):
    """Twin C51-style critics: logits per atom, shared discrete support (see td3dist / FastSAC)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        if num_atoms < 3:
            raise ValueError("num_atoms must be > 2 for distributional Q.")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        critic_input_dim = obs_dim + act_dim
        self.critic_1 = nn.Sequential(
            ResidualMLP([critic_input_dim, 512, 512, 512], activation),
            nn.Linear(512, num_atoms),
        )
        self.critic_2 = nn.Sequential(
            ResidualMLP([critic_input_dim, 512, 512, 512], activation),
            nn.Linear(512, num_atoms),
        )

        self.register_buffer(
            "q_support",
            torch.linspace(v_min, v_max, num_atoms),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.critic_1.apply(_init_sac_linear)
        self.critic_2.apply(_init_sac_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        z1 = self.critic_1(x)
        z2 = self.critic_2(x)
        return torch.cat([z1, z2], dim=-1)

    def get_value(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_pair = self(obs, act)
        evs = self.expected_values(logits_pair)
        return evs.mean(dim=-1, keepdim=True)

    def expected_values(self, logits_pair: torch.Tensor) -> torch.Tensor:
        """Expected Q under softmax for each twin: logits_pair [B, 2 * num_atoms] -> [B, 2]."""
        log1, log2 = logits_pair.chunk(2, dim=-1)
        e1 = expected_q_from_logits(log1, self.q_support)
        e2 = expected_q_from_logits(log2, self.q_support)
        return torch.cat([e1, e2], dim=-1)

    def bellman_projection(
        self,
        next_logits: torch.Tensor,
        rewards: torch.Tensor,
        discount: torch.Tensor | float,
    ) -> torch.Tensor:
        """Categorical projection (Bellman backup onto the fixed support)."""
        return ValueDistribution(next_logits, self.q_support).project(rewards, discount)



class TanhNormalActor(nn.Module):
    """Policy trunk + Gaussian + tanh squash (same layout as blade_runner SAC)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_norm: str = None,
        std_max: float = 0.5,
        std_min: float | None = None, # keep for future use
        action_init: Literal["zeros", "orthogonal"] = "zeros",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.trunk = MLP([obs_dim, 256, 256, 256], nn.SiLU, layer_norm=layer_norm)
        self.action = nn.Linear(256, act_dim * 2)
        self.trunk.apply(_init_sac_linear)
        
        if action_init == "orthogonal":
            self.action.apply(lambda m: _init_sac_linear(m, gain=0.01))
        elif action_init == "zeros":
            # zero-init following FastSAC
            nn.init.constant_(self.action.weight, 0.0) # zero-init the weight
            nn.init.constant_(self.action.bias, 0.0) # zero-init the bias
        else:
            raise ValueError(f"Invalid action_init: {action_init}")
        
        self.upscale: torch.Tensor
        self.register_buffer("upscale", torch.ones(act_dim))
        
        if not std_max > 0.0:
            raise ValueError("std_max must be positive")
        self.log_std_max = math.log(std_max)

    def forward(self, obs: torch.Tensor):
        feat = self.trunk(obs)
        mean, raw = self.action(feat).chunk(2, dim=-1)
        log_std = self.log_std_max - F.softplus(raw)
        dist = ScaledTanhNormal(mean, torch.exp(log_std), upscale=self.upscale)
        return dist


class NormalActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_norm: str = None,
        pred_std: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_std = pred_std
        self.trunk = MLP([obs_dim, 256, 256, 256], nn.SiLU, layer_norm=layer_norm)
        if self.pred_std:
            self.action = nn.Linear(256, act_dim * 2)
            self.log_std_max = math.log(1.0)
        else:
            self.action = nn.Linear(256, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.trunk.apply(_init_sac_linear)
        self.action.apply(lambda m: _init_sac_linear(m, gain=0.01))
    
    def forward(self, obs: torch.Tensor):
        feat = self.trunk(obs)
        if self.pred_std:
            mean, raw = self.action(feat).chunk(2, dim=-1)
            log_std = self.log_std_max - F.softplus(raw)
            dist = IndependentNormal(mean, torch.exp(log_std))
        else:
            mean = self.action(feat)
            mean = torch.tanh(mean / 2.0) * 2.0
            log_std = torch.exp(self.log_std) * torch.ones_like(mean)
            dist = IndependentNormal(mean, log_std)
        return dist


class SAC(TensorDictModuleBase):
    def __init__(
        self,
        cfg: SACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.env = env

        fake = observation_spec.zero()
        obs_dim = fake[OBS_KEY].shape[-1]
        act_dim = action_spec.shape[-1]

        if self.cfg.vecnorm:
            self.vecnorm_obs = VecNorm(obs_dim, decay=1.0).to(device)
        else:
            self.vecnorm_obs = nn.Identity()

        if self.cfg.distributional:
            v_min, v_max = -1.0, 9.0
            num_atoms = int((v_max - v_min) / 0.05) + 1
            self.Q = TwinDistributionalQNetwork(
                obs_dim,
                act_dim,
                num_atoms=num_atoms,
                v_min=v_min, # we actually do not have negative values, but it is a good idea to have a small margin
                v_max=v_max 
            ).to(device)
            self.V = nn.Identity()  # unused; keeps optim / checkpoint layout stable
            self.V_quantile = 0.7
        else:
            self.Q = TwinQNetwork(obs_dim, act_dim, layer_norm=self.cfg.critic_vecnorm).to(device)
            self.V = nn.Sequential(
                MLP([obs_dim, 512, 512], nn.SiLU),
                nn.Linear(512, 1),
            ).to(device)
            self.V.apply(_init_sac_linear)
            self.V_quantile = 0.7

        self.gae = GAE(self.cfg.gamma, self.cfg.gae_lambda).to(device)
        self.actor = TanhNormalActor(
            obs_dim,
            act_dim,
            layer_norm=self.cfg.actor_vecnorm,
            std_max=0.9,
            action_init=self.cfg.actor_init,
        ).to(device)

        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.Q_target.requires_grad_(False)
        self.actor_target.requires_grad_(False)

        if self.cfg.target_entropy_sigma is not None:
            self.target_entropy = gaussian_target_entropy(
                act_dim, self.cfg.target_entropy_sigma
            )
        else:
            self.target_entropy = -float(act_dim)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=device))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_Q = torch.optim.Adam(self.Q.parameters(), lr=self.cfg.lr)
        # self.opt_V = torch.optim.Adam(self.V.parameters(), lr=self.cfg.lr)

        self.global_step = 0

        if env is None:
            raise ValueError("SAC requires env for ReplayBuffer layout (fake_tensordict).")
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), "collector")
            .detach()
            .cpu()
        )
        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.actor.act_dim)
        self.rb = ReplayBuffer(self.cfg.buffer_size, fake_rb)
        self.msr = (
            MultiStepReturn(self.cfg.gamma, self.cfg.n_steps).to(device)
            if self.cfg.n_steps > 1
            else None
        )
        if self.cfg.actor_loss == "sac":
            self.actor_loss_fn = SACLoss(behavior_coef=self.cfg.actor_behavior_coef)
        elif self.cfg.actor_loss in ("advantage_weighted_regression", "weighted_regression"):
            self.actor_loss_fn = AdvantageWeightedRegression(
                num_candidates=self.cfg.wr_num_candidates,
                temperature=self.cfg.wr_temperature,
                normalize_advantage=self.cfg.wr_normalize_advantage,
            )
        else:
            raise ValueError(f"Unknown actor_loss: {self.cfg.actor_loss!r}")

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        """Train mode stochastic exploration; eval/deploy use deterministic Tanh-normal mean."""
        def policy(tensordict: TensorDict):
            obs = self.vecnorm_obs(tensordict[OBS_KEY])
            dist = self.actor(obs)
            action = dist.sample()
            tensordict[ACTION_KEY] = action
            tensordict["loc"] = dist.loc
            return tensordict
        return policy

    def on_stage_start(self, stage: str):
        self.enable_actor = True

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        self.global_step += self.cfg.train_every

        td = tensordict.exclude(("next", "stats"), "collector")
        reward = td[REWARD_KEY]
        # KEEP THIS FOR DEBUGGING
        if self.cfg.debug:
            # debug: constant reward scaled by effective horizon
            # the value should converge to 1.0 in this case
            # multi-step return should significantly speed up convergence
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)
            neg_rew_ratio = 0.0
        else:
            reward = reward.sum(-1, keepdim=True)
            neg_rew_ratio = (reward <= 0.).float().mean().item()
            reward = reward.clamp_min(0.)
        td[REWARD_KEY] = reward

        bs = td.batch_size
        # StackingCollector stacks steps on batch dim 1: [num_envs, horizon, …].
        if len(bs) >= 2:
            for ti in range(int(bs[1])):
                self.rb.push(td[:, ti].cpu())
        else:
            self.rb.push(td.cpu())

        infos: dict = {"rb_size": len(self.rb), "critic/neg_rew_ratio": neg_rew_ratio}
        if self.global_step < self.cfg.warm_up_steps:
            return infos

        for _ in range(self.cfg.train_every * self.cfg.utd_ratio):
            infos.update(self.train_critic())

        if self.enable_actor:
            for _ in range(self.cfg.train_every):
                infos.update(self.train_actor())

        # if self.global_step % self.cfg.v_update_every == 0:
        #     for _ in range(self.cfg.v_inner):
        #         infos.update(self.train_v())

        return dict(sorted(infos.items()))

    def train_critic(self):
        batch = self.rb.sample(
            batch_size=self.cfg.critic_batch_size,
            steps=self.cfg.n_steps
        ).to(self.device) # [T, N]

        reward = batch[REWARD_KEY]
        if not isinstance(reward, torch.Tensor):
            reward = sum(reward.values())

        if self.cfg.n_steps == 1:
            obs = batch[OBS_KEY]
            act = batch[ACTION_KEY]
            next_obs = batch["next", OBS_KEY]
            discount = self.cfg.gamma * (1.0 - batch[TERM_KEY].float())
        else:
            assert self.msr is not None
            obs = batch[OBS_KEY][0]
            act = batch[ACTION_KEY][0]
            next_obs, reward, discount = self.msr(
                batch["next", OBS_KEY],
                batch[ACTION_KEY],
                reward,
                batch[TERM_KEY],
                batch[DONE_KEY],
            )

        obs = self.vecnorm_obs(obs)
        next_obs = self.vecnorm_obs(next_obs)

        with torch.no_grad():
            dist = self.actor_target(next_obs)
            next_action = dist.sample()
            next_log_prob = dist.log_prob(next_action)
            target_action = next_action + torch.randn_like(next_action) * self.cfg.target_action_noise
            alpha = self.log_alpha.exp()
            lp = next_log_prob
            if lp.dim() == 1:
                lp = lp.unsqueeze(-1)
            if lp.shape != reward.shape:
                lp = lp.reshape_as(reward)

            if self.cfg.distributional:
                assert isinstance(self.Q, TwinDistributionalQNetwork)
                # Fold soft Bellman entropy into rewards, then categorical projection (FastSAC-style).
                adjusted_reward = reward + discount * self.cfg.entropy_bonus * (-alpha * lp)
                next_logits = self.Q_target(next_obs, target_action)
                n1, n2 = next_logits.chunk(2, dim=-1)
                p1 = self.Q_target.bellman_projection(n1, adjusted_reward, discount)
                p2 = self.Q_target.bellman_projection(n2, adjusted_reward, discount)
                z = self.Q_target.q_support.to(
                    device=p1.device, dtype=p1.dtype
                ).view(1, -1)
                ev1 = (p1 * z).sum(-1, keepdim=True)
                ev2 = (p2 * z).sum(-1, keepdim=True)
                target_dist = torch.where(ev1 < ev2, p1, p2)
            else:
                entropy_bonus = -alpha * lp
                if entropy_bonus.shape != reward.shape:
                    entropy_bonus = entropy_bonus.reshape_as(reward)
                target_qs = self.Q_target(next_obs, target_action)
                target_q = target_qs.mean(dim=-1, keepdim=True)
                q_target = reward + discount * (
                    target_q + self.cfg.entropy_bonus * entropy_bonus
                )
                target_dist = None

        qs: torch.Tensor = self.Q(obs, act)
        if self.cfg.distributional:
            q1, q2 = qs.chunk(2, dim=-1)
            log_p1 = F.log_softmax(q1, dim=-1).clamp(min=-30.0)
            log_p2 = F.log_softmax(q2, dim=-1).clamp(min=-30.0)
            q_loss = -(
                (target_dist * log_p1).sum(-1) + (target_dist * log_p2).sum(-1)
            ).mean()
        else:
            q_loss = (qs - q_target).square().sum(-1).mean()

        self.opt_Q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_Q.step()
        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_Q)

        # Optional: use expectile regression to estimate the value
        # v_pred = self.V(obs)
        # q_pred = qs.detach().max(dim=-1, keepdim=True).values
        # assert q_pred.shape == v_pred.shape
        # v_err = q_pred - v_pred
        # vf_sign = (v_err < 0).float()
        # vf_weight = (1 - vf_sign) * self.V_quantile + vf_sign * (1 - self.V_quantile)
        # vf_loss = (vf_weight * (v_err ** 2)).mean()

        # self.opt_V.zero_grad(set_to_none=True)
        # vf_loss.backward()
        # self.opt_V.step()

        if self.cfg.distributional:
            q_exp = self.Q.expected_values(qs.detach())
            q_val_mean = q_exp.mean().item()
            q_val_std = q_exp.std(dim=-1).mean().item()
        else:
            q_val_mean = qs.detach().mean().item()
            q_val_std = qs.detach().std(dim=-1).mean().item()

        return {
            "critic/q_loss": q_loss.item(),
            "critic/q_value": q_val_mean,
            "critic/q_std": q_val_std,
            # "critic/v_loss": vf_loss.item(),
            # "critic/v_value": v_pred.mean().item(),
            # "critic/v_err": v_err.mean().item(),
        }

    def train_actor(self):
        batch = self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1).to(
            self.device
        ) # [N,]

        obs = batch[OBS_KEY]
        obs = self.vecnorm_obs(obs)

        td_actor = TensorDict(
            {OBS_KEY: obs, ACTION_KEY: batch[ACTION_KEY]},
            batch_size=batch.batch_size,
        )
        policy_term, entropy_est, dist, action_update = self.actor_loss_fn.compute(
            td_actor, self.actor, self.Q
        )
        alpha = self.log_alpha.exp()
        actor_loss = (
            0.002 * (-entropy_est.reshape_as(policy_term)) + policy_term
        ).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss = -(alpha * (-entropy_est.detach() + self.target_entropy)).mean()
        alpha_loss.backward()
        self.opt_alpha.step()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.cfg.max_grad_norm
        )
        self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        # how much the action mean changes compared to that in the replay buffer
        mean_change = (dist.loc.detach() - batch["loc"]).abs().mean()
        infos = {
            "actor/loss": actor_loss.item(),
            "actor/mean_change": mean_change.item(),
            "actor/grad_norm": actor_grad_norm.item(),
            "actor/alpha": alpha.detach().item(),
            "actor/entropy": entropy_est.mean().item(),
        }
        actor_diagnostics = {}
        if isinstance(dist, ScaledTanhNormal):
            eps = 0.05
            with torch.no_grad():
                tanh_grad = 1.0 - (action_update.detach() / dist.upscale).square()
                action_saturation = (1.0 - action_update.detach().abs() / dist.upscale < eps)
                mean_squashed = torch.tanh(dist.loc.detach() / dist.upscale) * dist.upscale
                mean_saturation = (1.0 - mean_squashed.abs() / dist.upscale < eps)
                # mean saturation per action dimension
                dim_saturation = mean_saturation.float().mean(dim=0)
            actor_diagnostics = {
                "actor/action_saturation": action_saturation.float().mean().item(),
                "actor/mean_saturation": mean_saturation.float().mean().item(),
                "actor/max_saturation": dim_saturation.max().item(),
                "actor/tanh_grad": tanh_grad.mean().item(),
                "actor/tanh_grad_min": tanh_grad.min().item(),
                "actor/upscale": dist.upscale.mean().item(),
            }
            # self.actor.upscale.add_((dim_saturation > 0.1).float() * 3e-4)
        infos.update(actor_diagnostics)
        return infos

    def train_v(self):
        """On-policy-style V update: last `v_trace_steps` ring-buffer rows + GAE (ppo.common layout [N, T, …])."""
        if len(self.rb) <= self.cfg.v_trace_steps:
            return {}
        batch = self.rb.last(steps=self.cfg.v_trace_steps).to(self.device)

        reward = batch[REWARD_KEY]

        # Ring buffer layout: [T, N, …]. GAE expects [N, T, …].
        obs_tn = batch[OBS_KEY]
        next_obs_tn = batch["next", OBS_KEY]
        T, N = obs_tn.shape[:2]
        flat = T * N

        obs_tn = self.vecnorm_obs(obs_tn)
        next_obs_tn = self.vecnorm_obs(next_obs_tn)
        vals_tn = (
            self.V(obs_tn.reshape(flat, obs_tn.shape[-1])).reshape(T, N, 1)
        )
        next_vals_tn = (
            self.V(next_obs_tn.reshape(flat, next_obs_tn.shape[-1])).reshape(T, N, 1)
        )

        r_nt = reward.transpose(0, 1)
        term_nt = batch[TERM_KEY].transpose(0, 1).float()
        done_nt = batch[DONE_KEY].transpose(0, 1).float()
        val_nt = vals_tn.transpose(0, 1)
        next_val_nt = next_vals_tn.transpose(0, 1)

        with torch.no_grad():
            _, ret = self.gae(r_nt, term_nt, done_nt, val_nt, next_val_nt)

        pred_nt = vals_tn.transpose(0, 1)
        v_loss = F.mse_loss(pred_nt, ret)

        self.opt_V.zero_grad(set_to_none=True)
        v_loss.backward()
        self.opt_V.step()

        return {
            "critic/v_loss": v_loss.item(),
            "critic/v_value": pred_nt.mean().item(),
        }

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["Q"] = self.Q.state_dict()
        state_dict["V"] = self.V.state_dict()
        state_dict["actor"] = self.actor.state_dict()
        state_dict["Q_target"] = self.Q_target.state_dict()
        state_dict["actor_target"] = self.actor_target.state_dict()
        state_dict["opt_actor"] = self.opt_actor.state_dict()
        state_dict["opt_Q"] = self.opt_Q.state_dict()
        # state_dict["opt_V"] = self.opt_V.state_dict()
        state_dict["opt_alpha"] = self.opt_alpha.state_dict()
        state_dict["log_alpha"] = self.log_alpha.detach()
        state_dict["vecnorm_obs"] = self.vecnorm_obs.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.Q.load_state_dict(state_dict["Q"], strict=strict)
        self.V.load_state_dict(state_dict["V"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.Q_target.load_state_dict(state_dict["Q_target"], strict=strict)
        self.actor_target.load_state_dict(state_dict["actor_target"], strict=strict)
        self.opt_actor.load_state_dict(state_dict["opt_actor"])
        self.opt_Q.load_state_dict(state_dict["opt_Q"])
        # self.opt_V.load_state_dict(state_dict["opt_V"])
        self.opt_alpha.load_state_dict(state_dict["opt_alpha"])
        self.log_alpha.data = state_dict["log_alpha"].to(self.device)
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"])

