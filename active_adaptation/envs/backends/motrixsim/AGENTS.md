# AGENTS.md — working with the MotrixSim backend

Hard-won, non-obvious knowledge for this backend. Read before touching `motrixsim_sim.py`.
A **working reference** for motrixsim on legged robots lives at
`/home/ran/umi-on-LEGS/locomani_workspace/envs/motrix_wb_track/motrix_vec_env.py` — copy its
patterns when in doubt.

---

## ⭐ Go2 isaac→motrix policy transfer (2026-05-30) — READ THIS FIRST

Goal: take a Go2 loco policy trained in **isaac** and make it walk in this backend (sim2sim, no
retraining). It looked "genuinely wrong" (robot insane), and it WAS a bug, not a sim2sim gap. Full
write-up + diagnosis chain in the user memory `motrixsim-training-fix.md`. The transfer **works at
single-env now**; multi-env is still open (see bottom).

### Env-var flags you can set (the knobs we used to find/fix this)
| Flag | Default | Effect / when to use |
|---|---|---|
| `MOTRIX_FEET_ONLY=1` | off (**full-body collision is default**) | Opt INTO feet-only collision. **Don't** for Go2 — see "THE bug" below. |
| `NO_STEPN=1` | off (step_n on) | Honest per-substep stepping (`env.decimation≈9`) instead of step_n's `decimation=1`. **Needed for faithful transfer** — step_n's decimation=1 breaks the action filter + a multistep obs (see "decimation=1 bugs"). |
| `NO_GRAVITY_FF=1` | off (ff on) | Disable the gravity-comp feedforward. **Use it for transfer** — isaac has no gravity_ff and our explicit PD already droops the same as isaac (measured). ff is for native-training bootstrap only. |
| `KP_SCALE` / `KD_SCALE` | `1.0` | Multiply PD gains. For Go2 `kp=25,kd=0.5` already match isaac; leave at 1.0. (Did NOT fix multi-env collapse.) |
| `MOTRIX_ITERS` | `50` | Solver iteration count (Newton). 4 vs 200 made no difference to the multi-env issue. |
| `NO_MULTICCD=1` | off (multiccd on) | Disable the multiccd contact flag (debug; didn't change anything). |
| `GRAVITY_FF_DEBUG=1` | off | Print the measured gravity_ff magnitude. |

**Working transfer config:** `CUDA_VISIBLE_DEVICES="" NO_STEPN=1 NO_GRAVITY_FF=1 KD_SCALE=1`,
`task.num_envs=1`, full-body default. Result: upright 1.0, base_z 0.348 (target 0.35), 0 falls/450
steps, walks forward. (Isaac ground-truth zero-action stand = base_z 0.234, droop 0.41; motrix N=1 =
base_z 0.26, droop 0.34 → **the PDs match, no gravity_ff needed**.)

### THE bug — feet-only collision makes the Go2 tip and FLIP
The old default (feet-only: disable collision on non-`foot`/`ankle` bodies) makes Go2 an unstable
inverted pendulum on 4 point-feet (2.2 cm spheres). With the compliant explicit PD it tips and
**flips fully upside-down within ~1 s even at the default pose / zero action** (`projG_z` −1→+1, base
falls through the non-colliding floor to z=−0.3). No policy survives that. Isaac collides ALL link
geoms vs the ground, so feet-only was a divergence from isaac. **Fix = full-body collision** (now
default) + the umi-on-LEGS contact/solver recipe so the `LTL NotPositiveDefinite` blow-up that
originally motivated feet-only no longer happens:
- stiff **well-damped** contact `solref="-50000 -5000"` (damping −5000, NOT the −200/underdamped
  default — that chatter is what makes the solver go NotPositiveDefinite), `solimp="0.99 0.999 0.0001
  0.5 2"` on **every** collision geom **and** the floor;
- on the mujoco spec before compile: `solver=Newton`, `iterations`, `<flag multiccd>`; plus
  `model.options.max_iterations` after load;
- `contype=conaffinity=1` on all collision geoms — **MotrixSim has no MuJoCo contype/conaffinity
  OR-check**, both bits must be 1.
- Without the stiff floor `solref` the feet sink ~0.7 m THROUGH the floor (soft default contact).

### Secondary — step_n's `decimation=1` starves per-substep MDP machinery
step_n reports `physics_dt=step_dt` → `env.decimation=1`. That breaks three things, all only at
decimation==1, so isaac/mujoco (decimation≥2) are untouched:
1. **action alpha-EMA + delay** (`actions/joint.py:apply_action(substep)`) ticks ONCE/control-step vs
   isaac's 4 → the applied joint target is attenuated/lagged → weak gait / under-tracking.
2. **`joint_pos_multistep`** (obs): its 2-slot `joint_pos_substep` written at `substep%2` only fills
   slot 0 → `mean(1)=joint_pos/2` → **halved observation**.
3. **`joint_vel_multistep`** (obs): `diff` over a 1-sample buffer → NaN → poisons VecNorm at iter 0
   (already patched with a `shape<2` fallback to `asset.data.joint_vel`).
`NO_STEPN=1` (honest per-substep loop) sidesteps all three; it's the config that transfers. The
proper fix is to decouple the action filter's tick count from physics substeps (not done yet).

### OPEN — multi-env (N>1) collapse
N=1 walks; N≥16 **collapses** (base→0.1, droop→3.4) even passively, even with the active policy.
Identical-init envs stay bit-identical ~60 steps then diverge while collapsing together → the batched
contact solve is N-dependent and motrix's marginally-stable point-contact amplifies it. NOT fixed by:
spreading robots 10 m, solver iters (4/50/200), multiccd off, bigger feet, kp/kd, gravity_ff. umi
trains 4096 envs fine (cylinder feet + their tuning) so it's solvable. Leads: **capsule/cylinder
feet**, or **how the floor is attached for batching** (umi `robot_scene.attach(floor_scene).build()`
vs our add-geom-to-spec). This blocks native training; single-env demo/transfer is fine.

### Render the walking policy
Two-process Bevy viewer (torch + NVIDIA-Vulkan in one process segfaults, hence the split):
```
# process A (physics, publishes env-0 dof to /dev/shm):
CUDA_VISIBLE_DEVICES="" NO_STEPN=1 NO_GRAVITY_FF=1 ~/anaconda3/envs/env_isaaclab/bin/python \
  scripts/_viz_live_env.py backend=motrixsim task=Go2/Go2LocoFlat device=cpu \
  checkpoint_path=/tmp/isaac_walker_cpu.pt task.num_envs=1 +viz_fwd=1.0 '~task.randomization' headless=true &
# process B (Bevy window on the display, torch-free):
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json DISPLAY=:1 \
  ~/anaconda3/envs/env_isaaclab/bin/python scripts/_viz_live_render.py
```
Offline mp4 instead: `_viz_record.py` (rollout→npz) then `_viz_render.py` (npz→matplotlib mp4).
`/tmp/isaac_walker_cpu.pt` = the isaac Go2 walker, CPU-mapped (the sim2sim test checkpoint).

### Debug method that worked
sim2sim — run the isaac walker in the motrix env; a **zero-action stand test** (does it hold the
default pose?) and a **per-obs-term dump** localize bugs in seconds vs waiting for a train. `pkill -f`
matches the agent's own shell here (exit 144) — kill explicit PIDs instead.

---

## MotrixSim basics
- CPU-only (Rust), MJCF-native, **deterministic** stepping. Installed in conda env `env_isaaclab` (py3.11).
- Run everything with `CUDA_VISIBLE_DEVICES=""` (GPU is reserved for another task; motrixsim never needs it).
- model/data split: `m = mx.load_model(path)`, `d = mx.SceneData(m, batch=(N,))`, `m.step(d)` / `m.step_n(d, k)`.

## Reading state — SLICE dof_pos / dof_vel (do NOT use get_joint_dof_pos in the control path)
`dof_pos` layout = `[base_pos(3), base_quat(4, XYZW), joints(nj, qpos order)]`,
`dof_vel` = `[base_lin_vel(3, WORLD), base_ang_vel(3, BODY frame), joint_vel(nj, qvel order)]`.
- joints: `dof_pos[:, 7:7+nj]`, `dof_vel[:, 6:6+nj]`.
- base: pos `dof_pos[:,0:3]`, quat `dof_pos[:,3:7]` (**XYZW → convert to WXYZ**),
  lin vel `dof_vel[:,0:3]` (world), **ang vel `dof_vel[:,3:6]` is already BODY-frame**.
- **Gotcha that cost hours:** `body.get_joint_dof_pos(data)` *numerically equals* `dof_pos[:,7:]` for G1,
  but using it inside the PD path produced torques that diverged from the correct values once the robot
  moved (unexplained — likely a binding subtlety). **Slicing `dof_pos`/`dof_vel` fixed it.** Always slice.

## Control — PD-as-torque in NATIVE (qpos) order
g1.xml has `<motor>` actuators; send torque via `data.actuator_ctrls = tau` (shape (N, nu)).
- Compute PD entirely in native joint order: `tau = kp*(target - dof_pos[:,7:]) - kd*dof_vel[:,6:]`, clip to ctrlrange.
- The MDP layer here works in **Isaac order** (`joint_names_simulation`). Permute only at the boundary:
  reorder the action target Isaac→native once (`_mtx_from_isaac`), and reorder `applied_torque`/obs
  native→Isaac (`_jnt_mtx2isaac`). Do the PD math itself purely in native order.
- Actuators are 1:1 with joints in declaration order for G1, so `_act_jointidx` is identity (no scatter).

## Throughput — avoid per-link calls and per-substep Python
- **Do NOT** loop `link.get_linear_velocity()/get_angular_velocity()` over all links each substep — that
  throttled FPS badly. Get body velocities by **finite-differencing the single batched
  `get_link_poses(d)`** (override the root with exact `dof_vel`). See `update()`.
- The reference computes torque **once per env-step** and uses `m.step_n(d, decimation)` instead of a
  Python per-substep loop. Our `env_base` calls `write_data_to_sim`+`step`+`update` per substep, so the
  main lever here is keeping `update()` cheap (slicing + 1 pose call + finite-diff).

## Gravity compensation (reference technique; ours disabled)
Passive PD-to-default sags under gravity and tips a humanoid (~0.2 s). The reference fixes this with a
gravity-comp feedforward: legs = a **binary-searched** static offset (`kp*droop*scale` minimizing base
drift) measured during a settle; arms = dynamic `IkChain.get_bias_force(data)`. A naive
"settle-and-read-steady-state-torque" feedforward (`_setup_gravity_ff`) is **unstable** because the
settle itself droops/tips → bad measurement → it makes tipping worse. Port the binary-search/IkChain
version from `motrix_vec_env.py` (`_settle_for_obs_reference`, `_measure_gravity_offset_at_default`,
`_setup_arm_gravity_comp`) if you need it. For training-from-scratch the policy can learn balancing
without it; for sim2sim fidelity vs Isaac you'll likely need it.

## Robustness (already in place)
- `set_dof_pos` **validates the quaternion** — a diverged (NaN) env crashes it at reset. All write paths
  go through `_sanitize_dofs` (NaN scrub + quaternion renormalization).
- `MotrixSim.step` scrubs NaN, clamps runaway velocities (>200 rad/s), and has a `PanicException`
  recovery for `NotPositiveDefinite` solver blow-ups.

## API gotchas
- `d.set_dof_pos(arr, model)` (array THEN model). `d.set_dof_vel(arr)` (NO model arg — inconsistent!).
- `d.dof_pos`/`d.dof_vel` are read-only; subset indexing needs a **bool mask** (`d[mask]`), not int arrays.
- Contacts: `cq = m.get_contact_query(d); cq.is_colliding(pairs (P,2) uint32) -> (N,P) bool`. No per-body
  net force in the high-level API; g1.xml is feet-only collision (14 foot↔`floor` `<pair>`s).
- Quaternions are XYZW everywhere in motrixsim; this codebase uses WXYZ.

## Throughput reality (measured)
For **256 G1 envs on CPU**: raw `m.step` = **128 ms/step** (2000 substep-frames/s); `m.step_n(d,10)`
= **67 ms/substep** (~2x faster). Our backend's per-substep cost (write_data_to_sim + step + update +
callbacks) is ~**234 ms/substep → 121 env-frames/s**. So full humanoid loco (10–100M frames) is
~6–78 h on this CPU — NOT an overnight thing. Biggest levers, in order: (1) use `step_n` (compute
torque once per env-step, step all substeps in one call — needs a backend-specific substep loop since
`env_base` steps per-substep); (2) make `update()` cheap on intermediate substeps (only joint state is
needed for PD each substep; body poses/velocities only at the env-step end). For actual humanoid
*training*, a GPU backend (isaac/mjlab) is the practical choice; this backend's strength is
CPU sim2sim / contact-correctness validation.

## Status / next steps
See `RESULTS.md` and the **Go2 transfer section at the top** (the current state of the art for this
backend). The old "(a) torque divergence under large tip at passive-stand step ~59" was the
**contact instability** now understood: the robot tips because the contact/PD don't robustly hold it
— fixed for single-env by the full-body + stiff-damped-contact + Newton recipe; still the open
**multi-env collapse**. Remaining: (a) make N>1 stable (capsule feet / floor-attachment for batching);
(b) decouple the action-filter tick-count from physics substeps so `step_n` (fast) transfers as well
as `NO_STEPN`; (c) gravity comp (binary-search/IkChain) only if you go back to native training.
