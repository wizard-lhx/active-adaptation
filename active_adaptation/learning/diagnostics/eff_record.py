"""NPZ recording for effective impedance playbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .eff_config import ClampConfig


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


class Recorder:
    """Record one-environment impedance results and save every ten steps."""

    def __init__(
        self,
        path: Path,
        joint_names: list[str],
        physics_dt: float,
        control_dt: float,
        decimation: int,
        clamp: ClampConfig,
        augmented: bool,
        foot_names: list[str] | None = None,
    ) -> None:
        self.path = path
        self.joint_names = joint_names
        self.physics_dt = float(physics_dt)
        self.control_dt = float(control_dt)
        self.decimation = int(decimation)
        self.clamp = clamp
        self.augmented = augmented
        self.foot_names = foot_names
        if augmented:
            assert foot_names is not None
        self.data: dict[str, list[np.ndarray]] = {
            "steps": [],
            "Keff": [],
            "Deff": [],
            "Keff_eigvals": [],
            "Deff_eigvals": [],
            "Keff_cond": [],
            "Deff_cond": [],
            "kp": [],
            "kd": [],
        }
        if augmented:
            self.data.update(
                {
                    "Keff_aug": [],
                    "Deff_aug": [],
                    "D_base": [],
                    "J_leg": [],
                    "Jdot_valid": [],
                    "foot_vel_obs": [],
                    "foot_contact": [],
                }
            )
        if clamp.enabled:
            self.data.update(
                {
                    "tau_corr": [],
                    "joint_vel_substep": [],
                    "p_neg_substep": [],
                    "s_eigvals": [],
                    "delta_d_eigvals": [],
                    "clamp_applied": [],
                }
            )

    def record(
        self,
        step: int,
        impedance: dict[str, torch.Tensor],
        clamp_record: dict[str, Any] | None = None,
    ) -> None:
        self.data["steps"].append(np.asarray(step, dtype=np.int64))
        keys = (
            "Keff",
            "Deff",
            "Keff_eigvals",
            "Deff_eigvals",
            "Keff_cond",
            "Deff_cond",
            "kp",
            "kd",
        )
        for key in keys:
            self.data[key].append(_numpy(impedance[key][0]))
        if self.augmented:
            for key in (
                "Keff_aug",
                "Deff_aug",
                "D_base",
                "J_leg",
                "foot_vel_obs",
            ):
                self.data[key].append(_numpy(impedance[key][0]))
            self.data["Jdot_valid"].append(
                np.asarray(impedance["Jdot_valid"][0].item(), dtype=np.bool_)
            )
            self.data["foot_contact"].append(
                impedance["foot_contact"][0].detach().cpu().numpy().astype(np.bool_, copy=False)
            )
        if clamp_record is not None:
            clamp_keys = (
                "tau_corr",
                "joint_vel_substep",
                "p_neg_substep",
                "s_eigvals",
                "delta_d_eigvals",
            )
            for key in clamp_keys:
                self.data[key].append(_numpy(clamp_record[key]))
            self.data["clamp_applied"].append(np.asarray(clamp_record["clamp_applied"], dtype=np.bool_))
        if len(self.data["steps"]) == 1 or (len(self.data["steps"]) - 1) % 10 == 0:
            self.save()

    def save(self) -> None:
        payload = {key: np.stack(values) for key, values in self.data.items()}
        payload.update(
            {
                "joint_names": np.asarray(self.joint_names),
                "physics_dt": np.asarray(self.physics_dt, dtype=np.float32),
                "control_dt": np.asarray(self.control_dt, dtype=np.float32),
                "decimation": np.asarray(self.decimation, dtype=np.int64),
            }
        )
        if self.augmented:
            payload["foot_names"] = np.asarray(self.foot_names)
        if self.clamp.enabled:
            payload.update(
                {
                    "d_min": np.asarray(self.clamp.d_min, dtype=np.float32),
                    "tau_limit": np.asarray(self.clamp.tau_limit, dtype=np.float32),
                    "override_diag_c": np.asarray(self.clamp.override_diag_c, dtype=np.float32),
                }
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, **payload)
        temporary.replace(self.path)
