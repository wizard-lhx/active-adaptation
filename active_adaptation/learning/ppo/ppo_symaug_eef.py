"""ppo_symaug variant with effective-impedance evaluation metrics.

This file keeps the original ``ppo_symaug`` policy and ``scripts/train_ppo.py``
unchanged. Select it explicitly with ``algo=ppo_symaug_eef`` to add the
read-only effective impedance diagnostics implemented in ``ppo_eef.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import CMD_KEY, OBS_KEY
from active_adaptation.learning.ppo.ppo_eef import EffImpedanceConfig, EffImpedanceProbe
from active_adaptation.learning.ppo.ppo_symaug import PPOConfig as PPOBaseConfig
from active_adaptation.learning.ppo.ppo_symaug import PPOPolicy as PPOBasePolicy

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


@dataclass
class PPOEEFConfig(PPOBaseConfig):
    """Config for the ``ppo_symaug`` policy plus read-only impedance metrics."""

    _target_: str = "active_adaptation.learning.ppo.ppo_symaug_eef.PPOPolicy"
    name: str = "ppo_symaug_eef"
    eff_impedance: EffImpedanceConfig = field(default_factory=EffImpedanceConfig)
    eff_impedance_interval: int = 1


cs = ConfigStore.instance()
cs.store("ppo_symaug_eef", node=PPOEEFConfig, group="algo")


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return dict(cfg)


class PPOPolicy(PPOBasePolicy):
    """``ppo_symaug`` policy that appends effective impedance metrics to logs."""

    def __init__(self, cfg: PPOEEFConfig, *args, **kwargs) -> None:
        cfg_dict = _config_to_dict(cfg)
        eff_cfg = EffImpedanceConfig.from_any(cfg_dict.pop("eff_impedance", None))
        interval = int(cfg_dict.pop("eff_impedance_interval", 1))
        super().__init__(cfg_dict, *args, **kwargs)
        self.cfg = PPOEEFConfig(**cfg_dict, eff_impedance=eff_cfg, eff_impedance_interval=interval)
        self.eff_impedance_probe = EffImpedanceProbe(eff_cfg)
        self._eff_impedance_interval = max(interval, 1)
        self._eff_impedance_iter = 0
        self._eff_impedance_action_managers = []

    @classmethod
    def from_env(cls, cfg: PPOEEFConfig, env: "_EnvBase", device: str):
        policy = super().from_env(cfg, env, device)
        policy._eff_impedance_action_managers = list(policy._iter_action_managers(env.action_manager))
        policy._configure_eff_impedance_from_env(env)
        return policy

    def train_op(self, tensordict: TensorDict):
        diagnostic_logs = (
            self._compute_eff_impedance_logs(tensordict)
            if self.eff_impedance_probe.enabled
            else {}
        )
        infos = super().train_op(tensordict)
        infos.update(diagnostic_logs)
        return dict(sorted(infos.items()))

    def _compute_eff_impedance_logs(self, tensordict: TensorDict) -> dict[str, float]:
        iter_idx = self._eff_impedance_iter
        self._eff_impedance_iter += 1
        if iter_idx % self._eff_impedance_interval != 0:
            return {}
        kp, kd = self._eff_impedance_gains()
        self.eff_impedance_probe.sample_operating_points(self, tensordict, kp, kd)
        return self.eff_impedance_probe.compute_and_log({}, iter_idx)

    def _eff_impedance_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        kp_chunks = []
        kd_chunks = []
        for manager in self._eff_impedance_action_managers:
            joint_ids = manager.joint_ids
            if manager.action_dim != len(joint_ids):
                raise ValueError(
                    "effective impedance diagnostics require one action per controlled joint; "
                    f"got {manager!r}"
                )
            asset_data = manager.asset.data
            kp_chunks.append(asset_data.joint_stiffness[:, joint_ids].detach().mean(dim=0))
            kd_chunks.append(asset_data.joint_damping[:, joint_ids].detach().mean(dim=0))

        if not kp_chunks:
            raise ValueError("could not read joint_stiffness/joint_damping for impedance diagnostics")
        kp = torch.cat(kp_chunks, dim=-1)
        kd = torch.cat(kd_chunks, dim=-1)
        if kp.numel() != self.action_dim:
            raise ValueError(
                f"diagnostic gain dimension {kp.numel()} does not match action dim {self.action_dim}"
            )
        return kp, kd

    def _configure_eff_impedance_from_env(self, env: "_EnvBase") -> None:
        cfg = self.eff_impedance_probe.cfg
        controlled_joint_ids = self._eff_impedance_joint_ids()
        cfg.alpha = tuple(float(v) for v in self._eff_impedance_action_scaling().detach().cpu().tolist())
        cfg.q_slice = self._infer_joint_obs_slice(env, "pos", controlled_joint_ids)
        cfg.qd_slice = self._infer_joint_obs_slice(env, "vel", controlled_joint_ids)

        cfg_dict = _config_to_dict(self.cfg)
        cfg_dict.pop("eff_impedance", None)
        cfg_dict.pop("eff_impedance_interval", None)
        self.cfg = PPOEEFConfig(
            **cfg_dict,
            eff_impedance=cfg,
            eff_impedance_interval=self._eff_impedance_interval,
        )

    def _eff_impedance_joint_ids(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.as_tensor(manager.joint_ids, device=self.device)
                for manager in self._eff_impedance_action_managers
            ],
            dim=0,
        )

    def _eff_impedance_action_scaling(self) -> torch.Tensor:
        alpha_chunks = []
        for manager in self._eff_impedance_action_managers:
            alpha = torch.as_tensor(manager.action_scaling, device=self.device, dtype=torch.float32)
            if alpha.ndim == 0:
                alpha = alpha.expand(manager.action_dim)
            alpha_chunks.append(alpha.reshape(-1))
        alpha = torch.cat(alpha_chunks, dim=0)
        if alpha.numel() != self.action_dim:
            raise ValueError(
                f"diagnostic action scaling dimension {alpha.numel()} does not match action dim {self.action_dim}"
            )
        return alpha

    def _infer_joint_obs_slice(
        self,
        env: "_EnvBase",
        kind: str,
        controlled_joint_ids: torch.Tensor,
    ) -> tuple[int, int]:
        group = env.observation_funcs[OBS_KEY]
        actor_offset = 0
        if CMD_KEY in env.observation_spec.keys(True, True):
            actor_offset = env.observation_spec[CMD_KEY].shape[-1]

        local_offset = 0
        for obs in group.funcs.values():
            width = int(obs.compute().shape[-1])
            class_name = type(obs).__name__
            if self._matches_joint_obs_kind(class_name, kind):
                joint_ids = torch.as_tensor(obs.joint_ids, device=controlled_joint_ids.device)
                window = self._contiguous_joint_window(joint_ids, controlled_joint_ids)
                if window is not None:
                    start = actor_offset + local_offset + window
                    return (start, start + controlled_joint_ids.numel())
            local_offset += width

        raise ValueError(f"could not infer {kind} q slice from observation group {OBS_KEY!r}")

    @staticmethod
    def _matches_joint_obs_kind(class_name: str, kind: str) -> bool:
        if kind == "pos":
            return class_name in {"joint_pos", "joint_pos_multistep"}
        if kind == "vel":
            return class_name in {"joint_vel", "joint_vel_multistep"}
        raise ValueError(f"unknown joint observation kind {kind!r}")

    @staticmethod
    def _contiguous_joint_window(
        term_joint_ids: torch.Tensor,
        controlled_joint_ids: torch.Tensor,
    ) -> int | None:
        term_ids = [int(v) for v in term_joint_ids.detach().cpu().tolist()]
        controlled_ids = [int(v) for v in controlled_joint_ids.detach().cpu().tolist()]
        width = len(controlled_ids)
        for start in range(len(term_ids) - width + 1):
            if term_ids[start : start + width] == controlled_ids:
                return start
        return None

    @staticmethod
    def _iter_action_managers(action_manager):
        managers = getattr(action_manager, "action_managers", None)
        if managers is None:
            yield action_manager
            return
        for manager in managers:
            yield from PPOPolicy._iter_action_managers(manager)
