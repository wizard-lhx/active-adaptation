"""Closed-loop effective-damping clamp for play-time intervention."""

from __future__ import annotations

from typing import Any

import torch

from .eff_config import ClampConfig


class ClampController:
    """Apply a PSD damping correction through the effort-target channel."""

    def __init__(
        self,
        cfg: ClampConfig,
        policy: Any,
        env: Any,
        asset: Any,
        joint_ids: torch.Tensor,
    ) -> None:
        self.cfg = ClampConfig.from_any(cfg)
        assert env.num_envs == 1, "Effective-damping clamp requires task.num_envs=1."
        self.policy = policy
        self.asset = asset
        self.joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)

        state = asset.data.joint_vel[:, self.joint_ids]
        joints = self.joint_ids.numel()
        self._delta_d = state.new_zeros((1, joints, joints))
        self._s_eigvals = state.new_zeros((1, joints))
        self._s_eigvecs = torch.eye(joints, device=state.device, dtype=state.dtype).unsqueeze(0)
        self._delta_d_eigvals = state.new_zeros((1, joints))
        self._tau: list[torch.Tensor] = []
        self._qd: list[torch.Tensor] = []
        self._p_neg: list[torch.Tensor] = []

    def update(self, obs: Any) -> dict[str, torch.Tensor]:
        """Compute the impedance and correction for one control step."""

        result = self.policy.compute_impedance(obs)
        eigvals = result["Deff_eigvals"]
        eigvecs = result["Deff_eigvecs"]
        delta_eigvals = torch.relu(float(self.cfg.d_min) - eigvals)
        delta_d = (eigvecs * delta_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-2, -1)

        if self.cfg.override_diag_c > 0.0:
            eye = torch.eye(delta_d.shape[-1], device=delta_d.device, dtype=delta_d.dtype)
            self._delta_d = float(self.cfg.override_diag_c) * eye.unsqueeze(0)
        elif self.cfg.baseline_mode:
            self._delta_d = torch.zeros_like(delta_d)
        else:
            self._delta_d = delta_d

        self._s_eigvals = eigvals
        self._s_eigvecs = eigvecs
        self._delta_d_eigvals = delta_eigvals
        self._tau.clear()
        self._qd.clear()
        self._p_neg.clear()
        return result

    def substep_hook(self, substep: int) -> None:
        """Apply and record the correction at one physics substep."""

        del substep
        qd = self.asset.data.joint_vel[:, self.joint_ids]
        tau = -(self._delta_d @ qd.unsqueeze(-1)).squeeze(-1)
        tau = tau.clamp(-float(self.cfg.tau_limit), float(self.cfg.tau_limit))
        self.asset.set_joint_effort_target(tau, joint_ids=self.joint_ids)

        modal_vel = (self._s_eigvecs.transpose(-2, -1) @ qd.unsqueeze(-1)).squeeze(-1)
        negative = self._s_eigvals.clamp_max(0.0)
        p_neg = (negative * modal_vel.square()).sum(dim=-1)
        self._qd.append(qd[0].detach().clone())
        self._tau.append(tau[0].detach().clone())
        self._p_neg.append(p_neg[0].detach().clone())

    def step_record(self) -> dict[str, Any]:
        """Return the just-finished control step record."""

        return {
            "tau_corr": torch.stack(self._tau),
            "joint_vel_substep": torch.stack(self._qd),
            "p_neg_substep": torch.stack(self._p_neg),
            "s_eigvals": self._s_eigvals[0],
            "delta_d_eigvals": self._delta_d_eigvals[0],
            "clamp_applied": self.cfg.override_diag_c > 0.0 or not self.cfg.baseline_mode,
        }

    def zero_effort(self) -> None:
        """Clear the persistent effort target."""

        zero = torch.zeros_like(self.asset.data.joint_vel[:, self.joint_ids])
        self.asset.set_joint_effort_target(zero, joint_ids=self.joint_ids)
        self.asset.write_data_to_sim()
