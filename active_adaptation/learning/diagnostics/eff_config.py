"""Configuration for play-time effective impedance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClampConfig:
    enabled: bool = False
    baseline_mode: bool = False
    d_min: float = 0.0
    tau_limit: float = 50.0
    override_diag_c: float = 0.0

    @classmethod
    def from_any(cls, value: Any) -> "ClampConfig":
        if isinstance(value, cls):
            return value
        return cls() if value is None else cls(**dict(value))


@dataclass
class ImpedanceConfig:
    alpha: tuple[float, ...] | None = None
    q_slice: tuple[int, int] | None = None
    qd_slice: tuple[int, int] | None = None
    base_linvel_slice: tuple[int, int] | None = None
    base_angvel_slice: tuple[int, int] | None = None
    foot_vel_slice: tuple[int, int] | None = None
    clamp: ClampConfig = field(default_factory=ClampConfig)

    @property
    def augmented(self) -> bool:
        return None not in (
            self.base_linvel_slice,
            self.base_angvel_slice,
            self.foot_vel_slice,
        )

    @classmethod
    def from_any(cls, value: Any) -> "ImpedanceConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        data = dict(value)
        data["clamp"] = ClampConfig.from_any(data.get("clamp"))
        for key in (
            "alpha",
            "q_slice",
            "qd_slice",
            "base_linvel_slice",
            "base_angvel_slice",
            "foot_vel_slice",
        ):
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        return cls(**data)
