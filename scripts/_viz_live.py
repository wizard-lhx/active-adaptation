"""LIVE interactive MotrixSim viewer of the trained Go2 policy.

Opens a real Bevy window on the user's display (DISPLAY=:1) via
RenderApp(headless=False). The policy runs in a 32-env MotrixSim env (batch-1
collapses); we render ONLY env 0 by copying its state into a batch-1 render
scene each control step. The system camera stays active so the user can orbit
with the mouse. Auto-resets on fall, so it runs continuously.
"""
import os
import time
import tempfile
import hydra
import numpy as np
import torch
import mujoco
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
import motrixsim as mx

CONFIG_PATH = Path(__file__).parent.parent / "cfg"
GO2 = str(Path(aa.__file__).parent / "assets/Go2/mjcf/go2.xml")


def lookat_quat_wxyz(eye, target, up=(0, 0, 1)):
    eye = np.array(eye, float); target = np.array(target, float); up = np.array(up, float)
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    m = np.column_stack([r, u, -f]); tr = m.trace()
    if tr > 0:
        s = np.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return [w, x, y, z]


def build_render_model():
    """go2.xml + floor + umi-on-legs light + a robot-tracking camera (shown as the main view)."""
    spec = mujoco.MjSpec.from_file(GO2)
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.name = "floor"
    g.size = [0.0, 0.0, 0.05]; g.friction = [1.0, 0.1, 0.1]
    # exact umi-on-legs lighting (their injected MJCF light)
    light = spec.worldbody.add_light()
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = [0, 0, 4]; light.dir = [0, 0, -1]
    light.diffuse = [0.6, 0.6, 0.6]; light.specular = [0.2, 0.2, 0.2]
    # tracking camera (the system/orbit camera segfaults on mesh scenes on NVIDIA — umi-on-legs note)
    cam = spec.worldbody.add_camera()
    cam.name = "track"; cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    eye = (-1.3, -1.3, 0.8); cam.pos = list(eye); cam.quat = lookat_quat_wxyz(eye, (0, 0, 0.2))
    spec.compile()
    fd, p = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(GO2), prefix="_livemodel_")
    os.close(fd); open(p, "w").write(spec.to_xml())
    try:
        return mx.load_model(p)
    finally:
        os.remove(p)


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Open the Vulkan window FIRST, before warp/torch/the motrixsim env touch the GPU.
    # On NVIDIA, initializing those before Bevy's Vulkan init hard-crashes window creation;
    # initializing Vulkan first avoids the conflict. (The window tolerates the long env-build
    # gap with no sync.)
    rmodel = build_render_model()
    rdata = mx.SceneData(rmodel, batch=(1,))
    print("[live] opening window on DISPLAY=%s ..." % os.environ.get("DISPLAY"))
    app = mx.render.RenderApp(headless=False)
    app.__enter__()
    app.launch(rmodel, batch=1)
    app.system_camera.active = False                 # orbit camera segfaults on mesh scenes (NVIDIA)
    app.set_main_camera(rmodel.cameras.tolist()[0])  # show the robot-tracking camera instead
    app.sync(rdata)                                  # initial frame (default pose)

    aa.init(cfg, auto_rank=True)
    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.wandb import parse_checkpoint
    from active_adaptation.learning.modules.vecnorm import VecNorm

    env, policy = make_env_policy(cfg, parse_checkpoint(cfg.checkpoint_path))
    asset = env.scene.articulations["robot"]
    rollout_policy = policy.get_rollout_policy("eval")
    env.eval()
    VecNorm.FROZEN = True
    carry = env.reset()

    def sync_env0():
        dof0 = np.asarray(asset.mtx_data.dof_pos)[0:1].astype(np.float32).copy()
        rdata.set_dof_pos(dof0, rmodel)
        rmodel.forward_kinematic(rdata)
        app.sync(rdata)

    max_steps = int(cfg.get("viz_steps", 60000))
    print("[live] running — close the window or Ctrl-C to stop.")
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(max_steps):
            if getattr(app, "is_closed", False):
                print("[live] window closed."); break
            carry = rollout_policy(carry)
            _, carry = env.step_and_maybe_reset(carry)
            sync_env0()
    app.__exit__(None, None, None)
    print("[live] done.")


if __name__ == "__main__":
    main()
