"""Play a frozen policy while recording effective impedance."""

import copy
import datetime
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import hydra
import torch
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import active_adaptation as aa
from active_adaptation.learning.diagnostics.eff_clamp import ClampController
from active_adaptation.learning.diagnostics.eff_record import Recorder
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.utils.math import matrix_from_quat
from active_adaptation.utils.timerfd import Timer


DEFAULTS = [{"task": "Velocity"}, {"algo": "ppo_symaug_eff"}, "_self_"]

LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_JOINT_TYPES = ("hip", "thigh", "calf")
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in LEG_JOINT_TYPES
)


class GaitSymmetryStats:
    """Accumulate left-right load statistics during a single rollout."""

    def __init__(
        self,
        asset,
        contact_sensor,
        foot_contact_ids: torch.Tensor,
        leg_joint_ids: torch.Tensor,
        leg_joint_indices: torch.Tensor,
        physics_dt: float,
    ) -> None:
        self.asset = asset
        self.contact_sensor = contact_sensor
        self.foot_contact_ids = foot_contact_ids
        self.leg_joint_ids = leg_joint_ids
        self.leg_joint_indices = leg_joint_indices
        self.physics_dt = float(physics_dt)
        device = foot_contact_ids.device

        self.collecting = False
        self.in_stance = torch.zeros(4, device=device, dtype=torch.bool)
        self.stance_valid = torch.zeros(4, device=device, dtype=torch.bool)
        self.stance_impulse = torch.zeros(4, device=device)
        self.impulse_count = torch.zeros(4, device=device, dtype=torch.long)
        self.impulse_sum = torch.zeros(4, device=device)
        self.impulse_square_sum = torch.zeros(4, device=device)
        self.joint_torque_square_sum = torch.zeros(4, 3, device=device)
        self.torque_samples = 0
        self.keff_abs_sum = torch.zeros(4, 3, device=device)
        self.keff_aug_abs_sum = torch.zeros(4, 3, device=device)
        self.keff_samples = 0
        self.keff_aug_samples = 0

    def start_collection(self) -> None:
        """Start after warmup, excluding any stance already in progress."""
        contact = (
            self.contact_sensor.data.current_contact_time[0, self.foot_contact_ids]
            > 0.0
        )
        self.in_stance.copy_(contact)
        self.stance_valid.zero_()
        self.stance_impulse.zero_()
        self.collecting = True

    def substep_hook(self, _substep: int) -> None:
        if not self.collecting:
            return

        contact = (
            self.contact_sensor.data.current_contact_time[0, self.foot_contact_ids]
            > 0.0
        )
        force_z = self.contact_sensor.data.net_forces_w[
            0, self.foot_contact_ids, 2
        ].clamp_min(0.0)

        started = ~self.in_stance & contact
        ended = self.in_stance & ~contact
        completed = ended & self.stance_valid
        self.impulse_count.add_(completed)
        self.impulse_sum.add_(torch.where(completed, self.stance_impulse, 0.0))
        self.impulse_square_sum.add_(
            torch.where(completed, self.stance_impulse.square(), 0.0)
        )
        self.stance_impulse.copy_(
            torch.where(
                contact,
                self.stance_impulse + force_z * self.physics_dt,
                torch.zeros_like(self.stance_impulse),
            )
        )
        self.stance_valid.copy_((self.stance_valid | started) & contact)
        self.in_stance.copy_(contact)

        torque = self.asset.data.applied_torque[0, self.leg_joint_ids].reshape(4, 3)
        self.joint_torque_square_sum.add_(torque.square())
        self.torque_samples += 1

    def update_impedance(self, impedance: dict[str, torch.Tensor]) -> None:
        if not self.collecting:
            return

        keff_diag = torch.diagonal(impedance["Keff"][0]).index_select(
            0, self.leg_joint_indices
        )
        self.keff_abs_sum.add_(keff_diag.abs().reshape(4, 3))
        self.keff_samples += 1
        if "Keff_aug" in impedance:
            keff_aug_diag = torch.diagonal(impedance["Keff_aug"][0]).index_select(
                0, self.leg_joint_indices
            )
            self.keff_aug_abs_sum.add_(keff_aug_diag.abs().reshape(4, 3))
            self.keff_aug_samples += 1

    def discard_incomplete_stances(self) -> None:
        self.in_stance.zero_()
        self.stance_valid.zero_()
        self.stance_impulse.zero_()

    @staticmethod
    def _asymmetry(left: float, right: float) -> float:
        denominator = 0.5 * (left + right)
        if abs(denominator) <= 1e-12:
            return float("nan")
        return 100.0 * (left - right) / denominator

    @classmethod
    def _print_asymmetry(cls, name: str, values: list[float]) -> None:
        front = cls._asymmetry(values[0], values[1])
        rear = cls._asymmetry(values[2], values[3])
        left = 0.5 * (values[0] + values[2])
        right = 0.5 * (values[1] + values[3])
        side = cls._asymmetry(left, right)
        print(
            f"  {name} asymmetry [%]: front={front:+.1f} "
            f"rear={rear:+.1f} left-right={side:+.1f}"
        )

    @staticmethod
    def _format_legs(values: list[float]) -> str:
        return " ".join(
            f"{leg}={value:.3f}" for leg, value in zip(LEG_NAMES, values)
        )

    def print_summary(self, step: int) -> None:
        counts = self.impulse_count.float()
        valid_counts = counts.clamp_min(1.0)
        impulse_mean = self.impulse_sum / valid_counts
        impulse_var = (
            self.impulse_square_sum / valid_counts - impulse_mean.square()
        ).clamp_min(0.0)
        impulse_mean = torch.where(
            self.impulse_count > 0,
            impulse_mean,
            torch.full_like(impulse_mean, torch.nan),
        )
        impulse_std = torch.where(
            self.impulse_count > 0,
            impulse_var.sqrt(),
            torch.full_like(impulse_var, torch.nan),
        )
        joint_torque_rms = (
            self.joint_torque_square_sum / self.torque_samples
        ).sqrt()
        leg_torque_rms = self.joint_torque_square_sum.sum(-1).div(
            self.torque_samples * 3
        ).sqrt()

        impulse_mean_values = impulse_mean.tolist()
        impulse_std_values = impulse_std.tolist()
        impulse_count_values = self.impulse_count.tolist()
        leg_torque_values = leg_torque_rms.tolist()
        print(f"[gait-sym] step={step} cumulative after warmup")
        print(
            "  Jz mean+/-std [N*s/event]: "
            + " ".join(
                f"{leg}={mean:.3f}+/-{std:.3f}(n={count})"
                for leg, mean, std, count in zip(
                    LEG_NAMES,
                    impulse_mean_values,
                    impulse_std_values,
                    impulse_count_values,
                )
            )
        )
        self._print_asymmetry("Jz", impulse_mean_values)
        print(f"  leg torque RMS [N*m]: {self._format_legs(leg_torque_values)}")
        self._print_asymmetry("torque RMS", leg_torque_values)
        for joint_idx, joint_name in enumerate(LEG_JOINT_TYPES):
            print(
                f"  {joint_name} torque RMS [N*m]: "
                f"{self._format_legs(joint_torque_rms[:, joint_idx].tolist())}"
            )

        keff_mean = self.keff_abs_sum / self.keff_samples
        for joint_idx, joint_name in enumerate(LEG_JOINT_TYPES):
            print(
                f"  |Keff diag| {joint_name}: "
                f"{self._format_legs(keff_mean[:, joint_idx].tolist())}"
            )
        if self.keff_aug_samples:
            keff_aug_mean = self.keff_aug_abs_sum / self.keff_aug_samples
            for joint_idx, joint_name in enumerate(LEG_JOINT_TYPES):
                print(
                    f"  |Keff_aug diag| {joint_name}: "
                    f"{self._format_legs(keff_aug_mean[:, joint_idx].tolist())}"
                )


@dataclass
class IsaacAppConfig:
    headless: bool = "${..headless}"
    enable_cameras: bool = "${..record_video}"


@dataclass
class PlayTaskOverride:
    num_envs: int = 1
    max_episode_length: int = 5000
    record_video: bool = "${..record_video}"


@dataclass
class SymmetricEvalConfig:
    enabled: bool = False
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    action_alpha: float = 0.75


@dataclass
class PlayConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: HydraConf = field(default_factory=HydraConf)
    headless: bool = False
    backend: str = "isaac"
    device: str = "cuda"
    record_video: bool = False
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    seed: int = 42
    checkpoint_path: Optional[str] = None
    export_policy: bool = False
    discard_unused_obs: bool = True
    symmetric_eval: SymmetricEvalConfig = field(default_factory=SymmetricEvalConfig)
    disable_termination: bool = False
    gait_symmetry: bool = False
    gait_symmetry_print_interval: int = 300
    gait_symmetry_warmup_steps: int = 200
    task: PlayTaskOverride = field(default_factory=PlayTaskOverride)
    exploration_type: ExplorationType = ExplorationType.MODE


ConfigStore.instance().store(
    name="play_eff",
    node=PlayConfig(
        hydra=HydraConf(
            run=RunDir(dir="./outputs_play/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}")
        )
    ),
)


FILE_PATH = Path(__file__).parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _apply_symmetric_eval(cfg: PlayConfig) -> None:
    profile = cfg.symmetric_eval
    if not profile.enabled:
        return

    cfg.task.randomization = {
        "randomize_materials_isaac": {
            "body_names": ".*_foot",
            "static_friction_range": [profile.static_friction] * 2,
            "dynamic_friction_range": [profile.dynamic_friction] * 2,
            "restitution_range": [profile.restitution] * 2,
            "homogeneous": True,
        },
        "reset_joint_states_scale": {
            "pos_scales": {".*": [1.0, 1.0]},
        },
    }
    cfg.task.input.action.max_delay = 0
    cfg.task.input.action.alpha_range = [profile.action_alpha] * 2
    for group_cfg in cfg.task.observation.values():
        for observation_cfg in group_cfg.values():
            if OmegaConf.is_dict(observation_cfg) and "noise_std" in observation_cfg:
                observation_cfg.noise_std = 0.0

    print(
        "[symmetric-eval] "
        f"foot_friction=({profile.static_friction:g}, {profile.dynamic_friction:g}) "
        f"restitution={profile.restitution:g} action_delay=0 "
        f"action_alpha={profile.action_alpha:g} observation_noise=0"
    )


@VecNorm.freeze()
def export_policy(env, policy, export_dir):
    fake_input = env.observation_spec[0].rand().cpu().unsqueeze(0)
    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(deploy_policy, fake_input, str(export_dir / f"policy-{time_str}.onnx"))


@hydra.main(config_path=str(CONFIG_PATH), config_name="play_eff", version_base=None)
def main(cfg: PlayConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    _apply_symmetric_eval(cfg)
    if cfg.disable_termination:
        cfg.task.termination = {}
        print("[play] termination disabled; falls and timeouts will not reset the environment")
    aa.init(cfg, auto_rank=True)

    from active_adaptation.envs.utils import find_bodies, find_sensor_bodies
    from active_adaptation.helpers import make_env_policy

    env, policy = make_env_policy(
        task_cfg=cfg.task,
        algo_cfg=cfg.algo,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        discard_unused_obs=cfg.discard_unused_obs,
        checkpoint_path=cfg.checkpoint_path,
    )
    assert env.num_envs == 1, "Effective impedance recording requires task.num_envs=1."

    if cfg.export_policy:
        export_policy(env, policy, FILE_PATH / "exports" / str(cfg.task.name))

    stats_keys = [
        key for key in env.reward_spec.keys(True, True)
        if isinstance(key, tuple) and key[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval").to(env.device)
    base_env = env.base_env
    asset = base_env.scene.articulations["robot"]
    viewer_eye = torch.tensor(cfg.task.viewer.eye, device=env.device)
    viewer_target = torch.tensor(cfg.task.viewer.lookat, device=env.device)
    camera_distance = torch.linalg.vector_norm(viewer_eye[:2] - viewer_target[:2])
    camera_height = viewer_eye[2] - viewer_target[2]

    def reset_camera():
        root_position = asset.data.root_pos_w[0]
        heading = asset.data.heading_w[0]
        forward = torch.stack((heading.cos(), heading.sin(), torch.zeros_like(heading)))
        target = root_position.clone()
        target[2] += viewer_target[2]
        eye = target - camera_distance * forward
        eye[2] += camera_height
        base_env.sim.set_camera_view(eye=eye.cpu().tolist(), target=target.cpu().tolist())

    clamp_cfg = policy.impedance_cfg.clamp
    controller = None
    if clamp_cfg.enabled:
        controller = ClampController(
            clamp_cfg,
            policy,
            base_env,
            asset,
            policy.impedance_joint_ids(),
        )
        base_env._pre_step_callbacks.append(controller.substep_hook)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    output_path = run_dir / "eff_impedance" / "eff_impedance_timeseries.npz"
    augmented = policy.impedance_cfg.augmented

    if augmented or cfg.gait_symmetry:
        requested_foot_names = [f"{leg}_foot" for leg in LEG_NAMES]
        resolved_foot_body_ids, resolved_foot_names = find_bodies(
            asset,
            requested_foot_names,
        )
        body_id_by_name = dict(zip(resolved_foot_names, resolved_foot_body_ids))
        foot_names = requested_foot_names
        foot_body_ids = torch.as_tensor(
            [body_id_by_name[name] for name in foot_names],
            device=env.device,
            dtype=torch.long,
        )
        contact_sensor = base_env.scene.sensors["contact_forces"]
        resolved_contact_ids, resolved_contact_names = find_sensor_bodies(
            asset,
            contact_sensor,
            foot_names,
        )
        contact_id_by_name = dict(zip(resolved_contact_names, resolved_contact_ids))
        foot_contact_ids = torch.as_tensor(
            [contact_id_by_name[name] for name in foot_names],
            device=env.device,
            dtype=torch.long,
        )
    if augmented:
        joint_jacobian_ids = policy.impedance_joint_ids() + 6
    recorder = Recorder(
        output_path,
        policy.impedance_joint_names(),
        base_env.physics_dt,
        base_env.step_dt,
        base_env.decimation,
        clamp_cfg,
        augmented,
        foot_names if augmented else None,
    )
    previous_j_leg = None

    gait_stats = None
    if cfg.gait_symmetry:
        impedance_joint_ids = policy.impedance_joint_ids()
        impedance_joint_names = policy.impedance_joint_names()
        joint_index_by_name = {
            name: index for index, name in enumerate(impedance_joint_names)
        }
        leg_joint_indices = torch.as_tensor(
            [joint_index_by_name[name] for name in LEG_JOINT_NAMES],
            device=env.device,
            dtype=torch.long,
        )
        leg_joint_ids = impedance_joint_ids.index_select(0, leg_joint_indices)
        gait_stats = GaitSymmetryStats(
            asset,
            contact_sensor,
            foot_contact_ids,
            leg_joint_ids,
            leg_joint_indices,
            base_env.physics_dt,
        )
        base_env._post_step_callbacks.append(gait_stats.substep_hook)

    def compute_leg_jacobian():
        raw = asset.root_physx_view.get_jacobians()
        jacobian_w = (
            raw.index_select(1, foot_body_ids)
            .index_select(3, joint_jacobian_ids)[:, :, :3]
            .clone()
        )
        rotation_bw = matrix_from_quat(
            asset.data.root_link_quat_w
        ).transpose(-2, -1)
        return torch.matmul(rotation_bw.unsqueeze(1), jacobian_w)

    base_env.eval()
    carry = env.reset()
    reset_camera()
    timer = Timer(env.step_dt)
    video_enabled = bool(cfg.record_video)
    video_dir = run_dir / "videos"
    if video_enabled:
        video_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    video_path = video_dir / f"{cfg.task.name}-{time_str}.mp4"
    exploration = ExplorationType(cfg.exploration_type)
    last_print_time = time.perf_counter()
    last_print_step = -1

    with env.get_recorder(video_path, enabled=video_enabled) as video, set_exploration_type(exploration):
        try:
            for step in itertools.count():
                if gait_stats and step == cfg.gait_symmetry_warmup_steps:
                    gait_stats.start_collection()
                    print(
                        "[gait-sym] collection started after "
                        f"{cfg.gait_symmetry_warmup_steps} warmup control steps"
                    )
                with torch.inference_mode():
                    carry = rollout_policy(carry)
                diagnostic_obs = carry.clone().detach()
                if augmented:
                    j_leg = compute_leg_jacobian()
                    foot_contact = (
                        contact_sensor.data.current_contact_time[:, foot_contact_ids] > 0.0
                    )
                    jdot_valid = previous_j_leg is not None
                    jdot_leg = (
                        (j_leg - previous_j_leg) / base_env.step_dt
                        if jdot_valid
                        else torch.zeros_like(j_leg)
                    )
                else:
                    j_leg = None
                    jdot_leg = None
                    jdot_valid = False
                impedance = (
                    controller.update(diagnostic_obs, j_leg, jdot_leg)
                    if controller
                    else policy.compute_impedance(diagnostic_obs, j_leg, jdot_leg)
                )
                if gait_stats:
                    gait_stats.update_impedance(impedance)
                if augmented:
                    impedance["J_leg"] = j_leg
                    impedance["foot_contact"] = foot_contact
                    impedance["Jdot_valid"] = torch.full(
                        (env.num_envs,),
                        jdot_valid,
                        device=env.device,
                        dtype=torch.bool,
                    )
                    if step == 0:
                        print(
                            "[eff-aug] "
                            f"J_xe_norm={torch.linalg.vector_norm(impedance['J_xe']).item():.6g}"
                        )

                if video_enabled:
                    video.add_frame()
                with torch.inference_mode():
                    td, carry = env.step_and_maybe_reset(carry)
                episode_stats.add(td)

                clamp_record = controller.step_record() if controller else None
                recorder.record(step, impedance, clamp_record)
                done = td["next", "done"].any()
                if augmented:
                    previous_j_leg = None if done else j_leg.detach().clone()
                if done:
                    if controller:
                        controller.zero_effort()
                    if gait_stats:
                        gait_stats.discard_incomplete_stances()
                    reset_camera()

                if (
                    gait_stats
                    and gait_stats.collecting
                    and (step + 1) % cfg.gait_symmetry_print_interval == 0
                ):
                    gait_stats.print_summary(step + 1)

                if len(episode_stats) >= env.num_envs:
                    print("Step", step)
                    for key, value in sorted(episode_stats.pop().items(True, True)):
                        print(key, torch.mean(value).item())

                now = time.perf_counter()
                elapsed = now - last_print_time
                if elapsed >= 2.0:
                    steps = step - last_print_step
                    rate = steps / elapsed
                    print(f"step {step} | {rate:.1f} env steps/s")
                    last_print_time = now
                    last_print_step = step
                timer.sleep()
        except KeyboardInterrupt:
            print(f"Interrupted by user. Latest impedance data: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
