import torch
import torch.nn as nn
from jaxtyping import Float


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