"""Roll out the trained Go2 policy in MotrixSim and render with MotrixSim's
NATIVE renderer (umi-on-legs pattern), fully headless via software Vulkan.

Key recipe (no display needed):
  * VK_ICD_FILENAMES=<lavapipe>  -> software Vulkan, no GPU/X server.
  * a separate render-model (go2.xml + floor + tracking camera) whose camera has
    set_render_target("image", W, H) BEFORE launch.
  * RenderApp(headless=True).launch(model, batch=1).
  * each control step: copy env-0's dof_pos into the render data, forward_kinematic,
    app.sync(render_data) [SceneData sync positions the camera], camera.capture().

The policy runs in the real env with num_envs>=32 (motrixsim collapses at batch 1);
we render ONLY env 0 by copying its state into a batch-1 render scene.
"""
import os
import time
import tempfile
import hydra
import numpy as np
import torch
import mujoco
import imageio
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
import motrixsim as mx

CONFIG_PATH = Path(__file__).parent.parent / "cfg"
GO2 = str(Path(aa.__file__).parent / "assets/Go2/mjcf/go2.xml")


def lookat_quat_wxyz(eye, target, up=(0, 0, 1)):
    eye = np.array(eye, float); target = np.array(target, float); up = np.array(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.column_stack([r, u, -f])
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return [w, x, y, z]


def build_render_model(W, H, track=True, eye=(-1.2, -1.2, 0.8), look=(0, 0, 0.18)):
    """go2.xml + floor (matching MotrixScene) + lighting + an offscreen camera that tracks the robot."""
    spec = mujoco.MjSpec.from_file(GO2)
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.name = "floor"
    g.size = [0.0, 0.0, 0.05]; g.friction = [1.0, 0.1, 0.1]
    # go2.xml ships no lights -> scene renders nearly black. Use umi-on-legs' exact light.
    light = spec.worldbody.add_light()
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = [0, 0, 4]; light.dir = [0, 0, -1]
    light.diffuse = [0.6, 0.6, 0.6]; light.specular = [0.2, 0.2, 0.2]
    cam = spec.worldbody.add_camera()
    cam.name = "offscreen"
    cam.mode = (mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM if track
                else mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    cam.pos = list(eye)
    cam.quat = lookat_quat_wxyz(eye, look)
    spec.compile()
    fd, p = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(GO2), prefix="_rendmodel_")
    os.close(fd); open(p, "w").write(spec.to_xml())
    try:
        model = mx.load_model(p)
    finally:
        os.remove(p)
    model.cameras.tolist()[0].set_render_target("image", W, H)
    return model


def overlay(frame, info):
    """Draw a command-vs-actual palette in the top-left corner."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(im)
    lines = [
        "Go2  -  MotrixSim native render",
        f"t = {info['t']:5.2f}s",
        "          CMD    ACTUAL",
        f"vx :  {info['cvx']:+5.2f}  {info['avx']:+5.2f} m/s",
        f"vy :  {info['cvy']:+5.2f}  {info['avy']:+5.2f} m/s",
        f"yaw:  {info['cyaw']:+5.2f}  {info['ayaw']:+5.2f} rad/s",
        f"base height: {info['z']:.2f} m  (target 0.35)",
    ]
    pad = 6
    d.rectangle([0, 0, 270, 16 * len(lines) + 2 * pad], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        d.text((pad, pad + i * 16), ln, fill=(255, 255, 255))
    return np.asarray(im)


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.wandb import parse_checkpoint
    from active_adaptation.learning.modules.vecnorm import VecNorm

    env, policy = make_env_policy(cfg, parse_checkpoint(cfg.checkpoint_path))
    asset = env.scene.articulations["robot"]
    cm = env.base_env.command_manager
    W, H = int(cfg.get("viz_w", 720)), int(cfg.get("viz_h", 540))
    n_steps = int(cfg.get("viz_steps", 400))
    track = cfg.get("viz_cam", "track") == "track"

    # render model + headless renderer
    rmodel = build_render_model(W, H, track=track)
    rdata = mx.SceneData(rmodel, batch=(1,))
    app = mx.render.RenderApp(headless=True)
    app.__enter__()
    app.launch(rmodel, batch=1)
    app.system_camera.active = False
    rcam = app.get_camera(0); rcam.active = True

    rollout_policy = policy.get_rollout_policy("eval")
    env.eval()
    VecNorm.FROZEN = True
    carry = env.reset()
    _expl = {"mode": ExplorationType.MODE, "mean": ExplorationType.MEAN,
             "random": ExplorationType.RANDOM}[cfg.get("viz_explore", "mode")]

    force_cmd = cfg.get("viz_cmd", "natural")
    cm.resample_interval = 10 ** 9

    def inject(i):
        if force_cmd == "natural":
            return
        t = i * 0.02
        if force_cmd == "forward":
            vx, vy, yaw = 1.0, 0.0, 0.0
        elif force_cmd == "sequence":
            vx, vy, yaw = [(0, 0, 0), (1.0, 0, 0), (1.4, 0, 0),
                           (0.5, 0, 1.2), (0.5, 0, -1.2), (0, 0.6, 0)][int(t // 3) % 6]
        else:
            vx, vy, yaw = 0.0, 0.0, 0.0
        dev = cm.cmd_linvel_b.device
        cm.cmd_linvel_b[:] = torch.tensor([vx, vy, 0.0], device=dev)
        cm.next_command_linvel[:] = torch.tensor([vx, vy, 0.0], device=dev)
        cm.cmd_yawvel_b[:] = yaw

    def capture_frame():
        app.sync(rdata)
        task = rcam.capture()
        for _ in range(120):
            app.sync(rdata)
            img = task.take_image()
            if img is not None:
                return img.pixels.copy()
            time.sleep(0.002)
        return None

    frames = []
    nq = np.asarray(asset.mtx_data.dof_pos).shape[1]
    with torch.inference_mode(), set_exploration_type(_expl):
        for i in range(n_steps):
            inject(i)
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)

            # copy env-0 state into the render scene, then sync (positions camera)
            dof0 = np.asarray(asset.mtx_data.dof_pos)[0:1].astype(np.float32).copy()
            rdata.set_dof_pos(dof0, rmodel)
            rmodel.forward_kinematic(rdata)
            px = capture_frame()
            if px is None:
                print(f"[warn] frame {i} capture failed"); continue

            yaw = float(np.arctan2(
                2 * (asset.data.root_quat_w[0, 0] * asset.data.root_quat_w[0, 3]
                     + asset.data.root_quat_w[0, 1] * asset.data.root_quat_w[0, 2]),
                1 - 2 * (asset.data.root_quat_w[0, 2] ** 2 + asset.data.root_quat_w[0, 3] ** 2)).cpu())
            vw = asset.data.root_com_lin_vel_w[0].cpu().numpy()
            avx = np.cos(yaw) * vw[0] + np.sin(yaw) * vw[1]
            avy = -np.sin(yaw) * vw[0] + np.cos(yaw) * vw[1]
            info = dict(
                t=i * 0.02, cvx=float(cm.cmd_linvel_b[0, 0]), cvy=float(cm.cmd_linvel_b[0, 1]),
                avx=avx, avy=avy, cyaw=float(cm.cmd_yawvel_b[0]),
                ayaw=float(asset.data.root_com_ang_vel_w[0, 2].cpu()),
                z=float(asset.data.root_pos_w[0, 2].cpu()))
            frames.append(overlay(px[..., :3], info))

    out = Path(cfg.get("viz_out", "/tmp/go2_native.mp4"))
    imageio.mimsave(out, frames, fps=int(cfg.get("viz_fps", 25)))
    print(f"VIZ_NATIVE_DONE {len(frames)} frames -> {out}")
    app.__exit__(None, None, None)


if __name__ == "__main__":
    main()
