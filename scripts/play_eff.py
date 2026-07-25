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
from active_adaptation.utils.timerfd import Timer


DEFAULTS = [{"task": "Velocity"}, {"algo": "ppo_symaug_eff"}, "_self_"]


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
    recorder = Recorder(
        output_path,
        policy.impedance_joint_names(),
        base_env.physics_dt,
        base_env.step_dt,
        base_env.decimation,
        clamp_cfg,
    )

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
                with torch.inference_mode():
                    carry = rollout_policy(carry)
                diagnostic_obs = carry.clone().detach()
                impedance = controller.update(diagnostic_obs) if controller else policy.compute_impedance(diagnostic_obs)

                if video_enabled:
                    video.add_frame()
                with torch.inference_mode():
                    td, carry = env.step_and_maybe_reset(carry)
                episode_stats.add(td)

                clamp_record = controller.step_record() if controller else None
                recorder.record(step, impedance, clamp_record)
                if td["next", "done"].any():
                    if controller:
                        controller.zero_effort()
                    reset_camera()

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
