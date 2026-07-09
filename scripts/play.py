"""
This script is used to play and visualize a policy in the environment.
"""

import time
import torch
import hydra
import itertools
import datetime
import copy
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import OmegaConf
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.learning.modules.vecnorm import VecNorm


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "ppo"},
    "_self_",
]


@dataclass
class IsaacAppConfig:
    """Isaac Lab AppLauncher settings (resolved from parent config)."""

    headless: bool = "${..headless}"
    """Mirror ``headless``; passed to Isaac Lab's AppLauncher."""
    enable_cameras: bool = "${..record_video}"
    """Mirror ``record_video``; enables camera sensors when recording."""


@dataclass
class PlayTaskOverride:
    """Play-specific overrides merged into the selected task config."""

    num_envs: int = 4
    """Number of parallel environments (kept small for interactive playback)."""


@dataclass
class PlayConfig:
    """Hydra root config for policy playback and visualization."""

    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    """Hydra defaults list: task config, algo config, then this config."""
    hydra: HydraConf = field(default_factory=HydraConf)
    """Hydra runtime settings (output directory, etc.)."""
    headless: bool = False
    """Run with a visible GUI window (``false``) or headless (``true``)."""
    backend: str = "isaac"
    """Simulation backend: ``isaac``, ``mujoco``, ``mjlab``, or ``motrix``."""
    device: str = "cuda"
    """Torch device for policy inference (e.g. ``cuda``, ``cpu``)."""
    record_video: bool = False
    """Record an MP4 of the rollout (Isaac backend only)."""
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    """Backend-specific application launcher config."""
    seed: int = 42
    """Random seed (offset by local rank in distributed runs)."""
    checkpoint_path: Optional[str] = None
    """Path or WandB URI to a policy checkpoint; ``null`` starts from scratch."""
    export_policy: bool = False
    """Export the deploy policy to ONNX after loading."""
    discard_unused_obs: bool = True
    """Drop observation groups not listed in ``algo.in_keys``."""
    task: PlayTaskOverride = field(default_factory=PlayTaskOverride)
    """Task overrides applied on top of the selected task config."""
    exploration_type: ExplorationType = ExplorationType.MODE


cs = ConfigStore.instance()
cs.store(
    name="play",
    node=PlayConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_play/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}"
            )
        )
    )
)


FILE_PATH = Path(__file__).parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


@VecNorm.freeze()
def export_policy(env, policy, export_dir):
    fake_input = env.observation_spec[0].rand().cpu()
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"policy-{time_str}.onnx"
    export_onnx(deploy_policy, fake_input, str(path))


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: PlayConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)
    
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
    
    if cfg.export_policy:
        export_dir = FILE_PATH / "exports" / str(cfg.task.name)
        export_policy(env, policy, export_dir)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval").to(env.device)
    
    env.base_env.eval()
    carry = env.reset()
    
    assert not env.base_env.training

    timer = Timer(env.step_dt)

    # Velocity tracking diagnostics -------------------------------------------------
    # A valid sample is one env-step whose commanded xy speed is greater than 0.3 m/s.
    # mean_ratio: mean(||base_xy_velocity|| / ||command_xy_velocity||) over valid samples.
    # steps_ratio>0.4: fraction of valid samples whose speed ratio is greater than 0.4.
    # valid_steps: number of valid env-steps accumulated in the current print window.
    base_env = env.base_env
    robot = base_env.scene.articulations.get("robot")
    command_manager = base_env.command_manager
    vel_tracking_enabled = (
        robot is not None
        and hasattr(robot.data, "root_com_lin_vel_w")
        and hasattr(command_manager, "cmd_linvel_w")
    )
    vel_tracking_print_interval = 1000
    vel_ratio_sum = torch.zeros((), device=env.device)
    vel_ratio_good = torch.zeros((), device=env.device)
    vel_ratio_steps = torch.zeros((), device=env.device)

    def reset_vel_tracking_stats():
        vel_ratio_sum.zero_()
        vel_ratio_good.zero_()
        vel_ratio_steps.zero_()

    def update_vel_tracking_stats():
        if not vel_tracking_enabled:
            return
        cmd_speed = command_manager.cmd_linvel_w[:, :2].norm(dim=-1)
        valid = cmd_speed > 0.3
        base_speed = robot.data.root_com_lin_vel_w[:, :2].norm(dim=-1)
        ratio = base_speed[valid] / cmd_speed[valid].clamp_min(1e-6)
        vel_ratio_sum.add_(ratio.sum())
        vel_ratio_good.add_((ratio > 0.4).float().sum())
        vel_ratio_steps.add_(valid.float().sum())

    def print_vel_tracking_summary(step: int, *, force: bool = False):
        if not vel_tracking_enabled:
            return
        valid_steps = int(vel_ratio_steps.item())
        if valid_steps == 0 and not force:
            return
        if valid_steps:
            mean_ratio = (vel_ratio_sum / vel_ratio_steps).item()
            good_ratio = (vel_ratio_good / vel_ratio_steps).item()
        else:
            mean_ratio = 0.0
            good_ratio = 0.0
        print(
            f"[vel-track step={step}] mean_ratio={mean_ratio:.2f}  "
            f"steps_ratio>0.4={good_ratio:.0%}  "
            f"valid_steps={valid_steps}"
        )
        reset_vel_tracking_stats()

    # Optional video recording (Isaac backend only). This remains safe under
    # KeyboardInterrupt because the recorder is a context manager that flushes
    # buffered frames on exit.
    record_enabled = bool(cfg.get("record_video", False))
    video_dir = FILE_PATH / "videos"
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    video_path = video_dir / f"{cfg.task.name}-{time_str}.mp4"
    exploration_type = ExplorationType(cfg.get("exploration_type", "MODE"))

    print_interval_s = 2.0
    last_print_time = time.perf_counter()
    last_print_step = -1

    with env.get_recorder(video_path, enabled=record_enabled) as rec, \
        torch.inference_mode(), set_exploration_type(exploration_type):
        try:
            for i in itertools.count():
                update_vel_tracking_stats()
                if i % vel_tracking_print_interval == 0:
                    print_vel_tracking_summary(i + 1, force=True)
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                episode_stats.add(td)

                if record_enabled:
                    rec.add_frame()

                if len(episode_stats) >= env.num_envs:
                    print("Step", i)
                    for k, v in sorted(episode_stats.pop().items(True, True)):
                        print(k, torch.mean(v).item())

                now = time.perf_counter()
                elapsed = now - last_print_time
                if elapsed >= print_interval_s:
                    n_steps = i - last_print_step
                    sps = n_steps / elapsed
                    print(f"step {i} | {sps:.1f}x{env.num_envs}={sps*env.num_envs:.1f} env steps/s")
                    last_print_time = now
                    last_print_step = i

                timer.sleep()
        except KeyboardInterrupt:
            print(f"Interrupted by user, video saved to: {video_path}" if record_enabled else "Interrupted by user.")
        finally:
            print_vel_tracking_summary(i, force=False)
    
    env.close()


if __name__ == "__main__":
    main()
