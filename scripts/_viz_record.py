"""Roll out the trained Go2 policy in motrixsim (headless) and record real body
poses + command + actual velocity each control step to an npz for offline render."""
import hydra
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa

CONFIG_PATH = Path(__file__).parent.parent / "cfg"


def _q_yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.wandb import parse_checkpoint
    env, policy = make_env_policy(cfg, parse_checkpoint(cfg.checkpoint_path))
    asset = env.scene.articulations["robot"]
    cm = env.base_env.command_manager

    n_steps = int(cfg.get("viz_steps", 1400))
    rollout_policy = policy.get_rollout_policy("eval")
    env.eval()  # mirror canonical evaluate(): eval the WHOLE TransformedEnv (obs noise off, etc.)
    from active_adaptation.learning.modules.vecnorm import VecNorm
    VecNorm.FROZEN = True  # freeze obs normalization to the trained stats
    carry = env.reset()

    _expl = {"mode": ExplorationType.MODE, "mean": ExplorationType.MEAN,
             "random": ExplorationType.RANDOM}[cfg.get("viz_explore", "mode")]

    rec = {k: [] for k in
           ["body_pos", "root_pos", "root_quat", "cmd_lin_b", "cmd_yaw", "vel_w", "yawrate", "done"]}

    def grab(d):
        def g(name, default=None):
            v = getattr(asset.data, name, None)
            return v if v is not None else default
        body_pos = asset.data.body_pos_w[0].cpu().numpy()
        root_pos = asset.data.root_pos_w[0].cpu().numpy()
        root_quat = asset.data.root_quat_w[0].cpu().numpy()  # wxyz
        vel_w = g("root_com_lin_vel_w")
        vel_w = (vel_w[0].cpu().numpy() if vel_w is not None
                 else g("root_lin_vel_w")[0].cpu().numpy())
        angw = g("root_com_ang_vel_w")
        yawrate = float(angw[0, 2].cpu()) if angw is not None else float(
            g("root_com_ang_vel_b")[0, 2].cpu())
        cmd_lin_b = cm.cmd_linvel_b[0, :2].detach().cpu().numpy()
        cmd_yaw = float(cm.cmd_yawvel_b[0].detach().cpu())
        rec["body_pos"].append(body_pos)
        rec["root_pos"].append(root_pos)
        rec["root_quat"].append(root_quat)
        rec["cmd_lin_b"].append(cmd_lin_b)
        rec["cmd_yaw"].append(cmd_yaw)
        rec["vel_w"].append(vel_w)
        rec["yawrate"].append(yawrate)

    force_cmd = cfg.get("viz_cmd", "natural")
    cm.resample_interval = 10 ** 9  # disable random resampling when forcing

    def inject(i):
        if force_cmd == "natural":
            return
        t = i * 0.02
        if force_cmd == "forward":
            vx, vy, yaw = 1.0, 0.0, 0.0
        elif force_cmd == "sequence":
            phase = int(t // 3) % 6
            vx, vy, yaw = [(0, 0, 0), (1.0, 0, 0), (1.4, 0, 0),
                           (0.5, 0, 1.2), (0.5, 0, -1.2), (0, 0.6, 0)][phase]
        else:
            vx, vy, yaw = 0.0, 0.0, 0.0
        dev = cm.cmd_linvel_b.device
        cm.cmd_linvel_b[:] = torch.tensor([vx, vy, 0.0], device=dev)
        cm.next_command_linvel[:] = torch.tensor([vx, vy, 0.0], device=dev)
        cm.cmd_yawvel_b[:] = yaw

    with torch.inference_mode(), set_exploration_type(_expl):
        for i in range(n_steps):
            inject(i)
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            grab(td)
            dn = td.get(("next", "done")) if ("next", "done") in td.keys(True) else None
            done = bool(dn[0].any().item()) if dn is not None else False  # env 0 only
            rec["done"].append(done)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["body_names"] = np.array(list(asset.body_names))
    save_path = Path(cfg.get("viz_out", "/tmp/go2_viz.npz"))
    np.savez(save_path, **out)
    print(f"VIZ_RECORD_DONE steps={len(out['done'])} -> {save_path}")
    print("falls(done)=", int(out["done"].sum()),
          "mean_speed_cmd=%.2f" % np.linalg.norm(out["cmd_lin_b"], axis=1).mean())


if __name__ == "__main__":
    main()
