"""Isaac-only locomotion command with an interactive vertical support sling."""

from __future__ import annotations

import math

import torch
from typing_extensions import override

from .locomotion import Twist


class InteractiveTwist(Twist):
    """Add a keyboard-controlled vertical spring sling to ``Twist`` teleoperation."""

    supported_backends = ("isaac",)

    def __init__(
        self,
        *args,
        sling_body_name: str = "base_link",
        sling_height_rate: float = 0.25,
        sling_height_range: float = 0.8,
        sling_frequency_hz: float = 1.5,
        sling_damping_ratio: float = 0.7,
        sling_max_force_weight_ratio: float = 3.0,
        mouse_grab_force: float = 1.0,
        mouse_push_acceleration: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sling_body_name = sling_body_name
        self.sling_height_rate = sling_height_rate
        self.sling_height_range = sling_height_range
        self.sling_frequency_hz = sling_frequency_hz
        self.sling_damping_ratio = sling_damping_ratio
        self.sling_max_force_weight_ratio = sling_max_force_weight_ratio
        self.mouse_grab_force = mouse_grab_force
        self.mouse_push_acceleration = mouse_push_acceleration

    @override
    def _initialize(self, env) -> None:
        super()._initialize(env)
        assert self.num_envs == 1, "Interactive play requires task.num_envs=1."
        assert self.env.sim.has_gui(), "Interactive play requires headless=false."

        body_ids, body_names = self.asset.find_bodies(self.sling_body_name)
        assert len(body_ids) == 1, (
            f"sling_body_name={self.sling_body_name!r} must match exactly one body; "
            f"matched {body_names}."
        )
        self.sling_body_id = body_ids[0]

        # The sling owns UP/DOWN; the policy height command remains at its initial midpoint.
        self.key_mappings_height.clear()
        self.sling_enabled = False
        self._h_pressed = False
        self._sling_reference_height = torch.zeros(1, device=self.device)
        self._sling_target_height = torch.zeros(1, device=self.device)
        self._total_mass = torch.zeros(1, device=self.device)
        self._spring_stiffness = torch.zeros(1, device=self.device)
        self._spring_damping = torch.zeros(1, device=self.device)
        self._max_sling_force = torch.zeros(1, device=self.device)
        self._sling_force_w = torch.zeros(1, 1, 3, device=self.device)
        self._zero_torque_w = torch.zeros_like(self._sling_force_w)

        import carb.settings
        import omni.physx.bindings._physx as physx_bindings

        settings = carb.settings.get_settings()
        settings.set_bool(physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED, True)
        settings.set_bool(physx_bindings.SETTING_MOUSE_GRAB, True)
        settings.set_bool(physx_bindings.SETTING_MOUSE_GRAB_WITH_FORCE, True)
        settings.set_float(physx_bindings.SETTING_MOUSE_PICKING_FORCE, self.mouse_grab_force)
        settings.set_float(physx_bindings.SETTING_MOUSE_PUSH, self.mouse_push_acceleration)

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self.sling_enabled = False
        self._h_pressed = self.keyboard_manager.key_pressed["H"]

        self._total_mass.copy_(self.asset.root_physx_view.get_masses()[0].sum())
        omega = 2.0 * math.pi * self.sling_frequency_hz
        self._spring_stiffness.copy_(self._total_mass * omega**2)
        self._spring_damping.copy_(
            2.0 * self.sling_damping_ratio * self._total_mass * omega
        )
        self._max_sling_force.copy_(
            self.sling_max_force_weight_ratio * self._total_mass * 9.81
        )

    @override
    def pre_step(self, substep: int) -> None:
        keys = self.keyboard_manager.key_pressed
        h_pressed = keys["H"]
        if h_pressed and not self._h_pressed:
            self.sling_enabled = not self.sling_enabled
            if self.sling_enabled:
                height = self.asset.data.body_com_pos_w[0, self.sling_body_id, 2]
                self._sling_reference_height.copy_(height)
                self._sling_target_height.copy_(height)
        self._h_pressed = h_pressed

        if not self.sling_enabled:
            return

        height_direction = float(keys["UP"]) - float(keys["DOWN"])
        self._sling_target_height.add_(
            height_direction * self.sling_height_rate * self.env.physics_dt
        )
        self._sling_target_height.clamp_(
            self._sling_reference_height,
            self._sling_reference_height + self.sling_height_range,
        )

        body_height = self.asset.data.body_com_pos_w[0, self.sling_body_id, 2]
        body_vertical_velocity = self.asset.data.body_com_lin_vel_w[
            0, self.sling_body_id, 2
        ]
        force_z = (
            self._total_mass * 9.81
            + self._spring_stiffness * (self._sling_target_height - body_height)
            - self._spring_damping * body_vertical_velocity
        ).clamp_min_(0.0)
        force_z = torch.minimum(force_z, self._max_sling_force)

        self._sling_force_w.zero_()
        self._sling_force_w[0, 0, 2] = force_z[0]
        self.asset.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._sling_force_w,
            torques=self._zero_torque_w,
            body_ids=[self.sling_body_id],
            is_global=True,
        )

    @override
    def debug_draw(self) -> None:
        super().debug_draw()
        if not self.sling_enabled:
            return

        body_pos = self.asset.data.body_com_pos_w[:, self.sling_body_id]
        target_pos = body_pos.clone()
        target_pos[:, 2] = self._sling_target_height
        self.env.debug_draw.vector(
            body_pos,
            target_pos - body_pos,
            color=(0.2, 1.0, 0.2, 1.0),
        )
        self.env.debug_draw.point(target_pos, color=(0.2, 1.0, 0.2, 1.0), size=12.0)
        self.env.debug_draw.vector(
            body_pos,
            0.2 * self._sling_force_w[:, 0] / (self._total_mass * 9.81),
            color=(1.0, 0.5, 0.1, 1.0),
        )


__all__ = ["InteractiveTwist"]
