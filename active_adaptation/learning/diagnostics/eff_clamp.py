"""Closed-loop effective-damping clamp for play-time intervention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EffClampConfig:
    """Configuration for the play-time effective-damping clamp."""

    enabled: bool = False
    baseline_mode: bool = False
    d_min: float = 0.0
    tau_limit: float = 50.0

    @classmethod
    def from_any(cls, value: Any) -> "EffClampConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        return cls(**dict(value))


class ClampController:
    """Apply a PSD damping correction through Isaac's effort-target channel."""

    def __init__(
        self,
        cfg: EffClampConfig,
        policy: Any,
        env: Any,
        asset: Any,
        leg_joint_ids: torch.Tensor,
    ) -> None:
        from .eff_impedance import EffImpedanceProbe

        self.cfg = EffClampConfig.from_any(cfg)
        self.enabled = bool(self.cfg.enabled)
        if self.enabled:
            assert env.num_envs == 1, "Effective-damping clamp requires task.num_envs=1."

        self.policy = policy
        self.env = env
        self.asset = asset
        self.leg_joint_ids = torch.as_tensor(
            leg_joint_ids,
            device=env.device,
            dtype=torch.long,
        )
        self.eff_impedance_cfg = policy.eff_impedance_probe.cfg
        self.mean_net = EffImpedanceProbe._build_mean_net(policy)

        num_joints = self.leg_joint_ids.numel()
        state = asset.data.joint_vel[:, self.leg_joint_ids]
        self._delta_d_applied = state.new_zeros((1, num_joints, num_joints))
        self._s_eigvals = state.new_zeros((1, num_joints))
        self._s_eigvecs = torch.eye(num_joints, device=state.device, dtype=state.dtype).unsqueeze(0)
        self._delta_d_eigvals = state.new_zeros((1, num_joints))
        self._clamp_applied = not self.cfg.baseline_mode
        self._tau_corr: list[torch.Tensor] = []
        self._joint_vel_substep: list[torch.Tensor] = []
        self._p_neg_substep: list[torch.Tensor] = []

    def update(self, obs: Any) -> None:
        """Compute the damping correction for the current control observation."""

        from .eff_impedance import (
            EffImpedanceProbe,
            assemble_effective_impedance,
            compute_policy_jacobians,
        )

        actor_obs = EffImpedanceProbe._extract_actor_input(obs).clone().detach()
        kp = self.asset.data.joint_stiffness[:, self.leg_joint_ids]
        kd = self.asset.data.joint_damping[:, self.leg_joint_ids]
        jacs = compute_policy_jacobians(self.mean_net, actor_obs, self.eff_impedance_cfg)
        deff = assemble_effective_impedance(jacs, kp, kd, self.eff_impedance_cfg)["Deff"]

        symmetric_deff = 0.5 * (deff + deff.transpose(-2, -1))
        eigvals, eigvecs = torch.linalg.eigh(symmetric_deff)
        delta_d_eigvals = torch.relu(float(self.cfg.d_min) - eigvals)
        delta_d = (
            eigvecs
            * delta_d_eigvals.unsqueeze(-2)
        ) @ eigvecs.transpose(-2, -1)

        self._delta_d_applied = (
            torch.zeros_like(delta_d) if self.cfg.baseline_mode else delta_d
        )
        self._s_eigvals = eigvals
        self._s_eigvecs = eigvecs
        self._delta_d_eigvals = delta_d_eigvals
        self._tau_corr.clear()
        self._joint_vel_substep.clear()
        self._p_neg_substep.clear()

    def substep_hook(self, substep: int) -> None:
        """Apply and record the correction at one physics substep."""

        del substep
        joint_vel = self.asset.data.joint_vel[:, self.leg_joint_ids]
        tau_corr = -(
            self._delta_d_applied @ joint_vel.unsqueeze(-1)
        ).squeeze(-1)
        tau_corr = torch.clamp(
            tau_corr,
            min=-float(self.cfg.tau_limit),
            max=float(self.cfg.tau_limit),
        )
        self.asset.set_joint_effort_target(tau_corr, joint_ids=self.leg_joint_ids)

        modal_vel = (
            self._s_eigvecs.transpose(-2, -1) @ joint_vel.unsqueeze(-1)
        ).squeeze(-1)
        negative_eigvals = torch.minimum(
            self._s_eigvals,
            torch.zeros_like(self._s_eigvals),
        )
        p_neg = (negative_eigvals * modal_vel.square()).sum(dim=-1)

        self._joint_vel_substep.append(joint_vel[0].detach().clone())
        self._tau_corr.append(tau_corr[0].detach().clone())
        self._p_neg_substep.append(p_neg[0].detach().clone())

    def zero_effort(self) -> None:
        """Clear the persistent effort target and write it to the simulator."""

        zero_effort = torch.zeros_like(
            self.asset.data.joint_vel[:, self.leg_joint_ids]
        )
        self.asset.set_joint_effort_target(zero_effort, joint_ids=self.leg_joint_ids)
        self.asset.write_data_to_sim()

    def pop_step_record(self) -> dict[str, Any]:
        """Return and clear the just-finished control step's substep record."""

        record = {
            "tau_corr": torch.stack(self._tau_corr),
            "joint_vel_substep": torch.stack(self._joint_vel_substep),
            "p_neg_substep": torch.stack(self._p_neg_substep),
            "delta_d_eigvals": self._delta_d_eigvals[0].detach().clone(),
            "clamp_applied": self._clamp_applied,
        }
        self._tau_corr.clear()
        self._joint_vel_substep.clear()
        self._p_neg_substep.clear()
        return record
