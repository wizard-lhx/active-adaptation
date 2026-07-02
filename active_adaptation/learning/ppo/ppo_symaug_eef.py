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

    @staticmethod
    def _iter_action_managers(action_manager):
        managers = getattr(action_manager, "action_managers", None)
        if managers is None:
            yield action_manager
            return
        for manager in managers:
            yield from PPOPolicy._iter_action_managers(manager)
