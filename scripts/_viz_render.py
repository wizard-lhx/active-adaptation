"""Render a recorded Go2 rollout (npz) to mp4 with matplotlib (no GL).
Skeleton from real body poses + command-vs-actual palette + velocity arrows."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio

NPZ = sys.argv[1] if len(sys.argv) > 1 else "/tmp/go2_b32.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/go2_walk.mp4"
STRIDE = 2          # render every Nth control step (50Hz -> 25fps)
FPS = 25

d = np.load(NPZ, allow_pickle=True)
names = list(d["body_names"])
body = d["body_pos"]            # (T,19,3)
root = d["root_pos"]            # (T,3)
quat = d["root_quat"]          # (T,4) wxyz
cmd = d["cmd_lin_b"]           # (T,2) body-frame vx,vy
cmdyaw = d["cmd_yaw"]          # (T,)
velw = d["vel_w"]              # (T,3) world
yawr = d["yawrate"]            # (T,)
T = len(root)

ni = {n: i for i, n in enumerate(names)}
legs = [("FL", "tab:red"), ("FR", "tab:orange"), ("RL", "tab:blue"), ("RR", "tab:green")]
chains = []
for pre, col in legs:
    chain = [ni["base"], ni[f"{pre}_hip"], ni[f"{pre}_thigh"], ni[f"{pre}_calf"], ni[f"{pre}_foot"]]
    chains.append((chain, col))
head = [ni["base"], ni["Head_upper"], ni["Head_lower"]]


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


frames = []
trail = []
for t in range(0, T, STRIDE):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0], projection="3d")
    bp = body[t]
    cx, cy, cz = root[t]
    trail.append((cx, cy))

    # ground grid
    g = np.arange(-2, 2.01, 0.5)
    for gx in g:
        ax.plot([gx + round(cx), gx + round(cx)], [g[0] + round(cy), g[-1] + round(cy)], [0, 0], color="0.85", lw=0.6)
    for gy in g:
        ax.plot([g[0] + round(cx), g[-1] + round(cx)], [gy + round(cy), gy + round(cy)], [0, 0], color="0.85", lw=0.6)

    # skeleton
    segs, cols = [], []
    for chain, col in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            segs.append([bp[a], bp[b]]); cols.append(col)
    for a, b in zip(head[:-1], head[1:]):
        segs.append([bp[a], bp[b]]); cols.append("0.4")
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=3))
    # feet (red if in contact)
    for pre, _ in legs:
        fp = bp[ni[f"{pre}_foot"]]
        ax.scatter([fp[0]], [fp[1]], [fp[2]], c=("red" if fp[2] < 0.04 else "k"), s=18)
    ax.scatter([cx], [cy], [cz], c="k", s=40, marker="s")

    # velocity arrows from base (world frame), command=red, actual=green
    yaw = yaw_of(quat[t])
    cvx, cvy = cmd[t]
    cmd_w = np.array([np.cos(yaw) * cvx - np.sin(yaw) * cvy, np.sin(yaw) * cvx + np.cos(yaw) * cvy])
    ax.quiver(cx, cy, cz + 0.15, cmd_w[0], cmd_w[1], 0, color="red", lw=2.5, arrow_length_ratio=0.3)
    ax.quiver(cx, cy, cz + 0.15, velw[t, 0], velw[t, 1], 0, color="lime", lw=2.5, arrow_length_ratio=0.3)

    ax.set_xlim(cx - 0.6, cx + 0.6); ax.set_ylim(cy - 0.6, cy + 0.6); ax.set_zlim(0, 0.55)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(elev=14, azim=-60 + t * 0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()

    # ---- side panel: palette + top-down ----
    axp = fig.add_axes([0.66, 0.0, 0.34, 1.0]); axp.axis("off")
    # actual body-frame velocity
    avx = np.cos(yaw) * velw[t, 0] + np.sin(yaw) * velw[t, 1]
    avy = -np.sin(yaw) * velw[t, 0] + np.cos(yaw) * velw[t, 1]
    txt = (
        f"Go2  •  {os.environ.get('SIM_LABEL', 'MotrixSim')}  •  trained policy\n"
        f"t = {t*0.02:4.1f}s\n\n"
        "            CMD     ACTUAL\n"
        f"vx :   {cvx:+5.2f}   {avx:+5.2f}  m/s\n"
        f"vy :   {cvy:+5.2f}   {avy:+5.2f}  m/s\n"
        f"yaw:   {cmdyaw[t]:+5.2f}   {yawr[t]:+5.2f}  rad/s\n\n"
        f"base height: {cz:4.2f} m  (target 0.35)\n"
    )
    axp.text(0.05, 0.97, txt, va="top", ha="left", family="monospace", fontsize=10.5,
             transform=axp.transAxes)
    axp.text(0.05, 0.55, "red = commanded   green = actual", color="0.3", fontsize=9,
             transform=axp.transAxes)
    # top-down inset
    axt = fig.add_axes([0.70, 0.06, 0.26, 0.40])
    tr = np.array(trail)
    axt.plot(tr[:, 0], tr[:, 1], "0.6", lw=1)
    axt.plot(cx, cy, "ks", ms=5)
    axt.arrow(cx, cy, cmd_w[0] * 0.3, cmd_w[1] * 0.3, color="red", width=0.01, head_width=0.05)
    axt.arrow(cx, cy, velw[t, 0] * 0.3, velw[t, 1] * 0.3, color="lime", width=0.01, head_width=0.05)
    axt.set_xlim(cx - 1, cx + 1); axt.set_ylim(cy - 1, cy + 1); axt.set_aspect("equal")
    axt.set_title("top-down (X-Y)", fontsize=8); axt.set_xticks([]); axt.set_yticks([])

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    frames.append(img.copy())
    plt.close(fig)

imageio.mimsave(OUT, frames, fps=FPS)
print(f"RENDER_DONE {len(frames)} frames -> {OUT}")
