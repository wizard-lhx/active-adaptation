import torch
from typing import TYPE_CHECKING
from typing_extensions import override
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor as IsaacContactSensor
    from mjlab.sensor import ContactSensor as MjlabContactSensor
    from active_adaptation.envs.env_base import EnvBase

from .base import RewardV2
from active_adaptation.envs.utils import find_bodies, find_sensor_bodies
from active_adaptation.utils.math import quat_rotate_inverse


class max_swing_height(RewardV2):
    def __init__(self, weight: float, body_names: str, target_height: float):
        super().__init__(weight)
        self.body_names_pattern = body_names
        self.target_height = target_height

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.max_height = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.rew = torch.zeros(self.num_envs, 1, device=self.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=torch.bool, device=self.device
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        self.max_height[env_ids] = 0.0

    @override
    def update(self):
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        self.max_height = torch.maximum(self.max_height, feet_pos_w[..., 2])
        self.first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_contact_ids
        ]

        # Cast from above the feet so raised treads are visible even before clearance.
        ground_query_pos = feet_pos_w.clone()
        ground_query_pos[..., 2] += 10.0
        ground_height = self.env.get_ground_height_at(ground_query_pos)
        max_clearance = (self.max_height - ground_height).clamp(0.0, self.target_height)
        self.rew = (self.first_contact * max_clearance).sum(1, keepdim=True)
        self.max_height = torch.where(self.first_contact, 0.0, self.max_height)

    @override
    def _compute(self) -> torch.Tensor:
        active = (~self.command_manager.is_standing_env) & self.first_contact.any(
            dim=1, keepdim=True
        )
        return self.rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class feet_sliding(RewardV2):
    supported_backends = ("isaac", "mjlab", "motrix")

    def __init__(self, body_names: str, weight: float):
        super().__init__(weight)
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.contact_data = self.contact_sensor.data
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_data.current_contact_time[:, self.body_contact_ids]
            > self.env.physics_dt
        )
        if self.env.backend == "isaac":
            feet_speed = self.asset.data.body_com_lin_vel_w[:, self.body_ids].norm(dim=-1)
        elif self.env.backend in ("mjlab", "motrix"):
            feet_speed = self.asset.data.body_link_lin_vel_w[:, self.body_ids].norm(dim=-1)
        sliding = (in_contact * feet_speed).sum(dim=1)
        return -sliding.reshape(self.num_envs, 1)


class quadruped_trot(RewardV2):
    """Reward either (FL-RR) or (FR-RL) are in contact but not both."""

    def __init__(self, weight: float, body_names: str):
        super().__init__(weight)
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_contact_ids]
            > 0.005
        )
        FL_RR = in_contact[:, [0, 3]].all(dim=1)
        FR_RL = in_contact[:, [1, 2]].all(dim=1)
        rew = torch.logical_xor(FL_RR, FR_RL)
        active = ~self.command_manager.is_standing_env
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class feet_clearance(RewardV2):
    """Penalize foot pairs that are too close in the base-frame horizontal plane."""

    def __init__(self, body_names: str, weight: float, thres: float = 0.1):
        super().__init__(weight)
        self.body_names_pattern = body_names
        self.thres = thres

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.num_feet = len(self.body_ids)
        self.pair_indices = torch.triu_indices(
            self.num_feet, self.num_feet, offset=1, device=self.device
        )

    @override
    def _compute(self) -> torch.Tensor:
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            feet_pos_w - self.asset.data.root_link_pos_w.unsqueeze(1),
        )
        distances = (
            feet_pos_b[:, self.pair_indices[0], :2]
            - feet_pos_b[:, self.pair_indices[1], :2]
        ).norm(dim=-1)
        shortfall = (1.0 - distances / self.thres).clamp_min(0.0)
        return -shortfall.square().sum(dim=1, keepdim=True)


class feet_air_time(RewardV2):
    def __init__(
        self,
        body_names: str,
        thres: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.body_names_pattern = body_names
        self.thres = thres

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

        self.articulation_body_ids, self.body_names = find_bodies(
            self.asset, self.body_names_pattern
        )
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    @override
    def _compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        reward = ((last_air_time - self.thres) * first_contact).sum(1)
        active = ~self.command_manager.is_standing_env
        return reward.reshape(self.num_envs, 1), active


class feet_contact_count(RewardV2):
    supported_backends = ("isaac", "mjlab", "motrix")

    def __init__(self, body_names: str, weight: float):
        super().__init__(weight)
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]

        self.articulation_body_ids, self.body_names = find_bodies(
            self.asset, self.body_names_pattern
        )
        self.body_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )

    @override
    def _compute(self):
        self.first_contact = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)


class single_foot_contact(RewardV2):
    """Reward for single foot contact. Useful for bi-pedal locomotion."""

    def __init__(
        self,
        body_names: str,
        margin: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.body_names_pattern = body_names
        self.margin = margin

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.margin
        single_contact = torch.where(torch.sum(in_contact, dim=1) == 1, 0.0, -1.0)
        valid = ~self.command_manager.is_standing_env
        return single_contact.reshape(self.num_envs, 1), valid.reshape(self.num_envs, 1)
