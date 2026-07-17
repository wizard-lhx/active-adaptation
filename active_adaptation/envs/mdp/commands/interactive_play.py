"""Isaac-only locomotion command with an interactive four-point support sling."""

from __future__ import annotations

import math

import torch
from typing_extensions import override

from active_adaptation.utils.math import quat_rotate

from .locomotion import Twist


class InteractiveTwist(Twist):
    """Add a keyboard-controlled four-point spring sling to ``Twist`` teleoperation."""

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
        sling_half_extents: tuple[float, float, float] = (0.25, 0.14, 0.075),
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
        self.sling_half_extents = sling_half_extents
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
        half_x, half_y, half_z = self.sling_half_extents
        self._sling_corner_offsets_b = torch.tensor(
            [
                [
                    [half_x, half_y, half_z],
                    [half_x, -half_y, half_z],
                    [-half_x, half_y, half_z],
                    [-half_x, -half_y, half_z],
                ]
            ],
            device=self.device,
        )
        self._sling_corner_offsets_w = torch.zeros(1, 4, 3, device=self.device)
        self._sling_corner_pos_w = torch.zeros(1, 4, 3, device=self.device)
        self._sling_target_pos_w = torch.zeros(1, 4, 3, device=self.device)
        self._sling_corner_forces_w = torch.zeros(1, 4, 3, device=self.device)
        self._sling_force_w = torch.zeros(1, 1, 3, device=self.device)
        self._sling_torque_w = torch.zeros_like(self._sling_force_w)

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
        self._sling_corner_forces_w.zero_()
        self._sling_force_w.zero_()
        self._sling_torque_w.zero_()

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
                height = self.asset.data.body_link_pos_w[0, self.sling_body_id, 2]
                self._sling_reference_height.copy_(height)
                self._sling_target_height.copy_(height)
        self._h_pressed = h_pressed

        if not self.sling_enabled:
            self._sling_corner_forces_w.zero_()
            self._sling_force_w.zero_()
            self._sling_torque_w.zero_()
            return

        height_direction = float(keys["UP"]) - float(keys["DOWN"])
        self._sling_target_height.add_(
            height_direction * self.sling_height_rate * self.env.physics_dt
        )
        self._sling_target_height.clamp_(
            self._sling_reference_height,
            self._sling_reference_height + self.sling_height_range,
        )

        body_pos_w = self.asset.data.body_link_pos_w[:, self.sling_body_id]
        body_quat_w = self.asset.data.body_link_quat_w[:, self.sling_body_id]
        body_lin_vel_w = self.asset.data.body_link_lin_vel_w[:, self.sling_body_id]
        body_ang_vel_w = self.asset.data.body_link_ang_vel_w[:, self.sling_body_id]
        self._sling_corner_offsets_w.copy_(
            quat_rotate(
                body_quat_w[:, None].expand(-1, 4, -1),
                self._sling_corner_offsets_b,
            )
        )
        self._sling_corner_pos_w.copy_(
            body_pos_w[:, None] + self._sling_corner_offsets_w
        )
        corner_vel_w = body_lin_vel_w[:, None] + torch.linalg.cross(
            body_ang_vel_w[:, None].expand(-1, 4, -1),
            self._sling_corner_offsets_w,
            dim=-1,
        )
        self._sling_target_pos_w.copy_(self._sling_corner_pos_w)
        target_corner_height = (
            self._sling_target_height + self.sling_half_extents[2]
        )
        self._sling_target_pos_w[..., 2] = target_corner_height[:, None]

        force_z = 0.25 * (
            self._total_mass[:, None] * 9.81
            + self._spring_stiffness[:, None]
            * (target_corner_height[:, None] - self._sling_corner_pos_w[..., 2])
            - self._spring_damping[:, None] * corner_vel_w[..., 2]
        )
        force_z.clamp_min_(0.0)
        force_z = torch.minimum(force_z, 0.25 * self._max_sling_force[:, None])

        self._sling_corner_forces_w.zero_()
        self._sling_corner_forces_w[..., 2] = force_z
        self._sling_force_w.copy_(
            self._sling_corner_forces_w.sum(dim=1, keepdim=True)
        )
        self._sling_torque_w.copy_(
            torch.linalg.cross(
                self._sling_corner_offsets_w,
                self._sling_corner_forces_w,
                dim=-1,
            ).sum(dim=1, keepdim=True)
        )
        self.asset.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._sling_force_w,
            torques=self._sling_torque_w,
            body_ids=[self.sling_body_id],
            is_global=True,
        )

    @override
    def debug_draw(self) -> None:
        super().debug_draw()
        if not self.sling_enabled:
            return

        corner_pos = self._sling_corner_pos_w.reshape(-1, 3)
        target_pos = self._sling_target_pos_w.reshape(-1, 3)
        self.env.debug_draw.vector(
            corner_pos,
            target_pos - corner_pos,
            color=(0.2, 1.0, 0.2, 1.0),
        )
        self.env.debug_draw.point(
            target_pos, color=(0.2, 1.0, 0.2, 1.0), size=12.0
        )
        self.env.debug_draw.vector(
            corner_pos,
            0.2
            * self._sling_corner_forces_w.reshape(-1, 3)
            / (0.25 * self._total_mass * 9.81),
            color=(1.0, 0.5, 0.1, 1.0),
        )


__all__ = ["InteractiveTwist"]
