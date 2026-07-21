"""Torch-only effective impedance calculation for position policies."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from active_adaptation.learning.modules import VecNorm
from active_adaptation.learning.ppo.common import CMD_KEY, OBS_KEY

from .eff_config import ImpedanceConfig


Tensor = torch.Tensor


def actor_input(obs: Tensor | TensorDictBase) -> Tensor:
    """Return the flattened input consumed by the PPO actor."""

    if torch.is_tensor(obs):
        return obs.reshape(-1, obs.shape[-1])
    if CMD_KEY in obs.keys(True, True):
        value = torch.cat((obs[CMD_KEY], obs[OBS_KEY]), dim=-1)
    else:
        value = obs[OBS_KEY]
    return value.reshape(-1, value.shape[-1])


def _vecnorm(policy: Any) -> VecNorm:
    for module in policy.vecnorm.modules():
        if isinstance(module, VecNorm):
            return module
    raise ValueError("Could not find VecNorm in policy.vecnorm")


def policy_mean(policy: Any, obs: Tensor) -> Tensor:
    """Evaluate the unsampled action mean without changing policy state."""

    single = obs.ndim == 1
    value = obs.unsqueeze(0) if single else obs
    normed = _vecnorm(policy)._normalize(value)
    td = TensorDict(
        {"_obs_normed": normed},
        batch_size=value.shape[:-1],
        device=value.device,
    )
    actor = policy.actor
    if isinstance(actor, nn.parallel.DistributedDataParallel):
        actor = actor.module
    mean = actor.get_dist_params(td)["loc"]
    return mean.squeeze(0) if single else mean


def _indices(bounds: tuple[int, int], device: torch.device) -> Tensor:
    return torch.arange(bounds[0], bounds[1], device=device)


def compute_jacobians(
    policy: Any,
    obs: Tensor,
    cfg: ImpedanceConfig,
) -> tuple[Tensor, Tensor]:
    """Compute d(mean)/dq and d(mean)/dqd for a batch of actor inputs."""

    q_ids = _indices(cfg.q_slice, obs.device)
    qd_ids = _indices(cfg.qd_slice, obs.device)
    q = obs.index_select(-1, q_ids)
    qd = obs.index_select(-1, qd_ids)

    def mean_single(base: Tensor, q_value: Tensor, qd_value: Tensor) -> Tensor:
        value = base.index_copy(0, q_ids, q_value)
        value = value.index_copy(0, qd_ids, qd_value)
        return policy_mean(policy, value)

    jacobian = torch.func.jacrev(mean_single, argnums=(1, 2))
    jq, jqd = torch.func.vmap(jacobian)(obs, q, qd)
    return jq.detach(), jqd.detach()


def _gain_rows(value: Any, batch: int, joints: int, ref: Tensor) -> Tensor:
    gain = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
    if gain.ndim == 0:
        return gain.expand(batch, joints)
    if gain.ndim == 1:
        return gain.unsqueeze(0).expand(batch, joints)
    return gain.expand(batch, joints)


def matrix_metrics(keff: Tensor, deff: Tensor) -> dict[str, Tensor]:
    """Return symmetric spectra and condition numbers on the input device."""

    symmetric_k = 0.5 * (keff + keff.transpose(-2, -1))
    symmetric_d = 0.5 * (deff + deff.transpose(-2, -1))
    deff_eigvals, deff_eigvecs = torch.linalg.eigh(symmetric_d)
    return {
        "Keff_eigvals": torch.linalg.eigvalsh(symmetric_k).detach(),
        "Deff_eigvals": deff_eigvals.detach(),
        "Deff_eigvecs": deff_eigvecs.detach(),
        "Keff_cond": torch.linalg.cond(symmetric_k).detach(),
        "Deff_cond": torch.linalg.cond(symmetric_d).detach(),
    }


def compute_impedance(
    policy: Any,
    obs: Tensor | TensorDictBase,
    kp: Tensor,
    kd: Tensor,
    cfg: ImpedanceConfig,
) -> dict[str, Tensor]:
    """Compute effective stiffness and damping at the supplied observations."""

    value = actor_input(obs).detach()
    jq, jqd = compute_jacobians(policy, value, cfg)
    batch, joints, _ = jq.shape
    kp_rows = _gain_rows(kp, batch, joints, jq)
    kd_rows = _gain_rows(kd, batch, joints, jq)
    alpha_rows = _gain_rows(cfg.alpha, batch, joints, jq)
    eye = torch.eye(joints, device=jq.device, dtype=jq.dtype).expand(batch, -1, -1)
    keff = kp_rows.unsqueeze(-1) * (eye - alpha_rows.unsqueeze(-1) * jq)
    deff = kd_rows.unsqueeze(-1) * eye - kp_rows.unsqueeze(-1) * alpha_rows.unsqueeze(-1) * jqd
    result = {
        "Keff": keff.detach(),
        "Deff": deff.detach(),
        "kp": kp_rows.detach(),
        "kd": kd_rows.detach(),
    }
    result.update(matrix_metrics(keff, deff))
    return result
