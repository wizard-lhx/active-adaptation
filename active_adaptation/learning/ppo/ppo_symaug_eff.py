"""ppo_symaug policy with play-time effective impedance support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
from hydra.core.config_store import ConfigStore

from active_adaptation.learning.diagnostics.eff_config import ImpedanceConfig
from active_adaptation.learning.diagnostics.eff_impedance import compute_impedance
from active_adaptation.learning.ppo.common import CMD_KEY, OBS_KEY
from active_adaptation.learning.ppo.ppo_symaug import PPOConfig as BaseConfig
from active_adaptation.learning.ppo.ppo_symaug import PPOPolicy as BasePolicy

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


@dataclass
class PPOConfig(BaseConfig):
    _target_: str = "active_adaptation.learning.ppo.ppo_symaug_eff.PPOPolicy"
    name: str = "ppo_symaug_eff"
    eff_impedance: ImpedanceConfig = field(default_factory=ImpedanceConfig)


ConfigStore.instance().store("ppo_symaug_eff", node=PPOConfig, group="algo")


def _config_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return dict(cfg)


class PPOPolicy(BasePolicy):
    """ppo_symaug with environment-derived impedance parameters."""

    def __init__(self, cfg: PPOConfig, *args, **kwargs) -> None:
        data = _config_dict(cfg)
        impedance_cfg = ImpedanceConfig.from_any(data.pop("eff_impedance", None))
        super().__init__(data, *args, **kwargs)
        self.cfg = PPOConfig(**data, eff_impedance=impedance_cfg)
        self.impedance_cfg = impedance_cfg
        self._impedance_managers = []

    @classmethod
    def from_env(cls, cfg: PPOConfig, env: "_EnvBase", device: str):
        policy = super().from_env(cfg, env, device)
        policy._impedance_managers = list(policy._iter_managers(env.action_manager))
        policy._configure_impedance(env)
        return policy

    def compute_impedance(self, obs: Any) -> dict[str, torch.Tensor]:
        kp, kd = self.impedance_gains()
        return compute_impedance(self, obs, kp, kd, self.impedance_cfg)

    def impedance_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        kp = []
        kd = []
        for manager in self._impedance_managers:
            kp.append(manager.asset.data.joint_stiffness[:, manager.joint_ids])
            kd.append(manager.asset.data.joint_damping[:, manager.joint_ids])
        return torch.cat(kp, dim=-1), torch.cat(kd, dim=-1)

    def impedance_joint_ids(self) -> torch.Tensor:
        return torch.cat(
            [torch.as_tensor(manager.joint_ids, device=self.device) for manager in self._impedance_managers]
        )

    def impedance_joint_names(self) -> list[str]:
        return [name for manager in self._impedance_managers for name in manager.joint_names]

    def _configure_impedance(self, env: "_EnvBase") -> None:
        joint_ids = self.impedance_joint_ids()
        self.impedance_cfg.alpha = tuple(self._action_scaling().detach().cpu().tolist())
        self.impedance_cfg.q_slice = self._joint_obs_slice(env, "pos", joint_ids)
        self.impedance_cfg.qd_slice = self._joint_obs_slice(env, "vel", joint_ids)
        self.cfg.eff_impedance = self.impedance_cfg

    def _action_scaling(self) -> torch.Tensor:
        scaling = []
        for manager in self._impedance_managers:
            value = torch.as_tensor(manager.action_scaling, device=self.device, dtype=torch.float32)
            scaling.append(value.expand(manager.action_dim).reshape(-1))
        return torch.cat(scaling)

    def _joint_obs_slice(
        self,
        env: "_EnvBase",
        kind: str,
        controlled_ids: torch.Tensor,
    ) -> tuple[int, int]:
        group = env.observation_groups[OBS_KEY]
        actor_offset = (
            env.observation_spec[CMD_KEY].shape[-1]
            if CMD_KEY in env.observation_spec.keys(True, True)
            else 0
        )
        local_offset = 0
        names = {"pos": {"joint_pos", "joint_pos_multistep"}, "vel": {"joint_vel", "joint_vel_multistep"}}
        for obs in group.funcs.values():
            width = int(obs.compute().shape[-1])
            if type(obs).__name__ in names[kind]:
                term_ids = torch.as_tensor(obs.joint_ids, device=controlled_ids.device)
                window = self._joint_window(term_ids, controlled_ids)
                if window is not None:
                    start = actor_offset + local_offset + window
                    return start, start + controlled_ids.numel()
            local_offset += width
        raise ValueError(f"could not infer {kind} slice from observation group {OBS_KEY!r}")

    @staticmethod
    def _joint_window(term_ids: torch.Tensor, controlled_ids: torch.Tensor) -> int | None:
        term = term_ids.tolist()
        controlled = controlled_ids.tolist()
        for start in range(len(term) - len(controlled) + 1):
            if term[start : start + len(controlled)] == controlled:
                return start
        return None

    @staticmethod
    def _iter_managers(manager):
        children = getattr(manager, "action_managers", None)
        if children is None:
            yield manager
        else:
            for child in children:
                yield from PPOPolicy._iter_managers(child)
