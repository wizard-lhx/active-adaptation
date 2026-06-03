"""Visualization for the MotrixSim Go2 backend — one entry point, four modes.

Select the mode with ``+viz_mode=`` (it's a new key, so it needs the ``+``):
  render  Torch-free Bevy window; shows whatever pose is published to /dev/shm.
  train   Train the policy and publish env-0 to /dev/shm (watch with a `render` process).
  play    Run a trained checkpoint and publish env-0 to /dev/shm.
  record  Roll out a checkpoint headless and save body poses/commands to an .npz.

The live view is TWO processes: importing torch in the same process as the NVIDIA Vulkan
renderer segfaults the driver, so `render` must stay torch-free. `torch` / `active_adaptation`
are therefore imported lazily inside the train/play/record modes only, and `render` runs
without hydra (no config needed — it just opens a window and reads /dev/shm).

Example (Go2, faithful decimation-4 config):
  # Terminal A — physics, publishes env-0:
  CUDA_VISIBLE_DEVICES="" NO_STEPN=1 python scripts/viz.py +viz_mode=train \
      backend=motrixsim task=Go2/Go2LocoFlat device=cpu task.num_envs=256 \
      headless=true '~task.randomization' +task.sim.motrixsim_physics_dt=0.005
  # Terminal B — the window:
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json DISPLAY=:1 python scripts/viz.py +viz_mode=render
"""
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(Path(__file__).parent.parent / "cfg")  # match play.py (do NOT .resolve)
GO2_SCENE = str(ROOT / "active_adaptation/assets/Go2/mjcf/scene_viz.xml")
# Fast tmpfs IPC between the physics and render processes; fall back to the system temp
# dir where /dev/shm is absent (macOS/Windows) so the tool stays portable.
_SHM = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
DOF_FILE = os.path.join(_SHM, "go2_live_dof.npy")
DOF_TMP = os.path.join(_SHM, "go2_live_dof_tmp.npy")


# --------------------------------------------------------------------- render (torch-free)
def run_render():
    """Open an NVIDIA Bevy window and render the env-0 pose published to /dev/shm."""
    import motrixsim as mx  # torch-free; importing torch here would crash the Vulkan driver

    model = mx.load_model(GO2_SCENE)
    data = mx.SceneData(model, batch=(1,))
    dof = np.asarray(data.dof_pos).copy()
    dof[0, :7] = [0, 0, 0.35, 0, 0, 0, 1]  # neutral pose until the first publish arrives
    data.set_dof_pos(dof.astype(np.float32), model)
    model.forward_kinematic(data)

    print("[render] opening window on DISPLAY=%s" % os.environ.get("DISPLAY"), flush=True)
    app = mx.render.RenderApp(headless=False)
    app.__enter__()
    app.launch(model, batch=1)
    app.system_camera.active = False
    app.set_main_camera(model.cameras.tolist()[0])
    app.sync(data)
    print("[render] RUNNING — close the window to stop.", flush=True)
    try:
        while not getattr(app, "is_closed", False):
            try:
                d = np.load(DOF_FILE)  # (1, nq) float32 from a physics process
                data.set_dof_pos(d.astype(np.float32), model)
                model.forward_kinematic(data)
            except Exception:
                pass  # writer not ready / mid-write -> hold the last pose
            app.sync(data)
            time.sleep(0.02)
    finally:
        app.__exit__(None, None, None)


# --------------------------------------------------------------------- torch-mode helpers
def _publish_env0(asset):
    """Atomically publish env-0's dof_pos to /dev/shm for the render process."""
    dof = np.asarray(asset.mtx_data.dof_pos)[0:1].astype(np.float32)
    np.save(DOF_TMP, dof)
    os.replace(DOF_TMP, DOF_FILE)


def _force_command(cm, cmd, i):
    """Override the velocity command for play/record (forward / sequence / stand / natural)."""
    import torch

    if cmd == "natural":
        return
    if cmd == "sequence":  # cycle stand / fwd / fast / turns / strafe, 3 s each
        vx, vy, yaw = [(0, 0, 0), (1.0, 0, 0), (1.4, 0, 0),
                       (0.5, 0, 1.2), (0.5, 0, -1.2), (0, 0.6, 0)][int(i * 0.02 // 3) % 6]
    elif cmd == "forward":
        vx, vy, yaw = 1.0, 0.0, 0.0
    else:
        vx, vy, yaw = 0.0, 0.0, 0.0
    dev = cm.cmd_linvel_b.device
    cm.cmd_linvel_b[:] = torch.tensor([vx, vy, 0.0], device=dev)
    if hasattr(cm, "next_command_linvel"):
        cm.next_command_linvel[:] = torch.tensor([vx, vy, 0.0], device=dev)
    cm.cmd_yawvel_b[:] = yaw


def _build(cfg, checkpoint=False):
    """aa.init + make_env_policy. checkpoint=True loads cfg.checkpoint_path."""
    import active_adaptation as aa

    aa.init(cfg, auto_rank=True)
    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.wandb import parse_checkpoint

    ck = parse_checkpoint(cfg.get("checkpoint_path")) if checkpoint else None
    return make_env_policy(cfg, ck)


# --------------------------------------------------------------------- train (publishes live)
def run_train(cfg):
    import torch
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from active_adaptation.utils.helpers import EpisodeStats

    env, policy = _build(cfg)
    asset = env.scene.articulations["robot"]
    transitions = cfg.algo.get("store_transitions", True)
    obs_keys = list(env.observation_spec.keys(True, True))
    stats = EpisodeStats(
        [k for k in env.reward_spec.keys(True, True) if isinstance(k, tuple) and k[0] == "stats"],
        device=env.device,
    )
    carry = env.reset()

    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(carry, rollout_policy):
        data = []
        for _ in range(cfg.algo.train_every):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            _publish_env0(asset)  # live env-0 view
            td = td.exclude(*[k for k in td.keys(True, True) if isinstance(k, str) and k.startswith("_")])
            if not transitions:
                td["next"] = td["next"].exclude(*obs_keys)
            data.append(td.to(policy.device))
        data = torch.stack(data, dim=1)
        if not transitions:
            sv = data["state_value"]
            nsv = policy.compute_value(carry.copy())["state_value"]
            nsv = torch.cat([data["state_value"][:, 1:], nsv.unsqueeze(1)], dim=1)
            data["next", "state_value"] = torch.where(data["next", "done"], sv, nsv)
        return data, carry

    policy.on_stage_start("")
    rollout_policy = policy.get_rollout_policy("train", critic=not transitions)
    print("VIZ_TRAIN num_envs=%d device=%s" % (env.num_envs, env.device), flush=True)
    frames = 0
    for i in range(int(cfg.get("viz_iters", 100000))):
        t = time.perf_counter()
        data, carry = collect(carry, rollout_policy)
        stats.add(data)
        frames += data.numel()
        policy.train_op(data)
        if i % 10 == 0:
            print("iter %5d  frames %9d  fps %5.0f"
                  % (i, frames, data.numel() / (time.perf_counter() - t)), flush=True)


# --------------------------------------------------------------------- play (publishes live)
def run_play(cfg):
    import torch
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from active_adaptation.learning.modules.vecnorm import VecNorm

    env, policy = _build(cfg, checkpoint=True)
    asset = env.scene.articulations["robot"]
    cm = env.base_env.command_manager
    rollout_policy = policy.get_rollout_policy("eval")
    env.eval()
    VecNorm.FROZEN = True
    cm.resample_interval = 10 ** 9
    carry = env.reset()
    cmd = cfg.get("viz_cmd", "forward")
    dt = 1.0 / int(cfg.get("viz_fps", 50))
    print("VIZ_PLAY cmd=%s publishing env-0 to %s" % (cmd, DOF_FILE), flush=True)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(int(cfg.get("viz_steps", 60000))):
            t0 = time.time()
            _force_command(cm, cmd, i)
            carry = rollout_policy(carry)
            _, carry = env.step_and_maybe_reset(carry)
            _publish_env0(asset)
            slp = dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)


# --------------------------------------------------------------------- record (-> npz)
def run_record(cfg):
    import torch
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from active_adaptation.learning.modules.vecnorm import VecNorm

    env, policy = _build(cfg, checkpoint=True)
    asset = env.scene.articulations["robot"]
    cm = env.base_env.command_manager
    rollout_policy = policy.get_rollout_policy("eval")
    env.eval()
    VecNorm.FROZEN = True
    cm.resample_interval = 10 ** 9
    carry = env.reset()
    cmd = cfg.get("viz_cmd", "forward")
    rec = {k: [] for k in
           ("body_pos", "root_pos", "root_quat", "cmd_lin_b", "cmd_yaw", "vel_w", "yawrate", "done")}
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(int(cfg.get("viz_steps", 450))):
            _force_command(cm, cmd, i)
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            rec["body_pos"].append(asset.data.body_pos_w[0].cpu().numpy())
            rec["root_pos"].append(asset.data.root_pos_w[0].cpu().numpy())
            rec["root_quat"].append(asset.data.root_quat_w[0].cpu().numpy())  # wxyz
            rec["vel_w"].append(asset.data.root_com_lin_vel_w[0].cpu().numpy())
            rec["yawrate"].append(float(asset.data.root_com_ang_vel_w[0, 2].cpu()))
            rec["cmd_lin_b"].append(cm.cmd_linvel_b[0, :2].detach().cpu().numpy())
            rec["cmd_yaw"].append(float(cm.cmd_yawvel_b[0].detach().cpu()))
            dn = td.get(("next", "done")) if ("next", "done") in td.keys(True) else None
            rec["done"].append(bool(dn[0].any().item()) if dn is not None else False)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["body_names"] = np.array(list(asset.body_names))
    path = cfg.get("viz_out", "/tmp/go2_viz.npz")
    np.savez(path, **out)
    print("VIZ_RECORD steps=%d falls=%d -> %s" % (len(out["done"]), int(out["done"].sum()), path), flush=True)


_TORCH_MODES = {"train": run_train, "play": run_play, "record": run_record}


@hydra.main(config_path=CONFIG_PATH, config_name="play", version_base=None)
def _main_torch(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    mode = cfg.get("viz_mode")
    if mode not in _TORCH_MODES:
        raise SystemExit("viz_mode must be one of: render, %s" % ", ".join(_TORCH_MODES))
    _TORCH_MODES[mode](cfg)


if __name__ == "__main__":
    # `render` runs without hydra/torch; everything else goes through the hydra config.
    # (viz_mode is a new key, so it's passed as `+viz_mode=...` — match either form.)
    if any("viz_mode=render" in a for a in sys.argv):
        run_render()
    else:
        # importing active_adaptation registers the hydra config search path (algo/, etc.)
        # and pulls in torch — fine for the train/play/record modes, not for render.
        import active_adaptation  # noqa: F401
        _main_torch()
