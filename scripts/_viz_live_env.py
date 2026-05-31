"""LIVE env-writer (process A): runs the trained Go2 policy in motrixsim and
publishes env-0's dof_pos to /dev/shm each step for the renderer process
(B = _viz_live_render.py). Kept SEPARATE because importing torch in the render
process segfaults the NVIDIA Vulkan driver, so the GPU renderer must be torch-free.
"""
import os
import time
import hydra
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa

CONFIG_PATH = Path(__file__).parent.parent / "cfg"
DOF_FILE = "/dev/shm/go2_live_dof.npy"
TMP = "/dev/shm/go2_live_dof_tmp.npy"


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
    rollout_policy = policy.get_rollout_policy("eval")
    env.eval()
    VecNorm.FROZEN = True
    carry = env.reset()

    force_fwd = float(cfg.get("viz_fwd", 0.0))  # if >0, hold a steady forward command
    if force_fwd > 0:
        cm.resample_interval = 10 ** 9
        if hasattr(cm, "use_stiffness"):
            cm.use_stiffness[:] = False
        if hasattr(cm, "fixed_yaw_speed"):
            cm.fixed_yaw_speed[:] = 0.0

    max_steps = int(cfg.get("viz_steps", 60000))
    dt = 1.0 / int(cfg.get("viz_fps", 50))
    print("[env] publishing env-0 dof to %s (%d steps)" % (DOF_FILE, max_steps), flush=True)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(max_steps):
            t0 = time.time()
            if force_fwd > 0:
                cm.cmd_linvel_b[:] = torch.tensor([force_fwd, 0.0, 0.0], device=cm.cmd_linvel_b.device)
                if hasattr(cm, "next_command_linvel"):
                    cm.next_command_linvel[:] = torch.tensor([force_fwd, 0.0, 0.0], device=cm.cmd_linvel_b.device)
                cm.cmd_yawvel_b[:] = 0.0
            carry = rollout_policy(carry)
            _, carry = env.step_and_maybe_reset(carry)
            dof = np.asarray(asset.mtx_data.dof_pos)[0:1].astype(np.float32)
            np.save(TMP, dof)
            os.replace(TMP, DOF_FILE)  # atomic publish (no torn reads)
            slp = dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
    print("[env] done.", flush=True)


if __name__ == "__main__":
    main()
