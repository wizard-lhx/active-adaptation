# MotrixSim backend — integration results (honest status)

Overnight work integrating **MotrixSim (Motphys) 0.6.1** as a 4th simulation backend
(`isaac` / `mujoco` / `mjlab` / **`motrixsim`**), validating contact correctness, and
attempting G1 humanoid locomotion training. **All CPU** (`CUDA_VISIBLE_DEVICES=""`) — GPU never touched.

## TL;DR
- ✅ MotrixSim wired in as a full backend (scene/sim adapters, batched data layer, asset converter).
- ✅ Contact correctness validated vs the `mujoco` backend on the identical MJCF (strong agreement).
- ✅ Training runs **end-to-end without crashing** (after fixing 2 bugs + adding NaN/divergence guards).
- ⚠️ **RL does NOT yet demonstrate clear learning** — episode length plateaus ~22 steps (0.44s),
     episode return declines. Root cause not fully resolved (see Open Issues). Treat the backend as
     **mechanically working but not yet validated for training**.

## What is solid
- Construction / reset / step / obs / reward / termination all run; G1 stands & feet contact correctly.
- **Contact correctness** (MuJoCo ref vs MotrixSim, same g1.xml, free-settle): base-z tracks <1cm
  through the fall; **foot-contact onset identical (0.116s)**. `crash` termination ≈0 while `fall_over`
  fires — consistent with the **feet-only collision model** (only 14 foot↔`floor` `<pair>`s), matching
  the mujoco backend (and unlike Isaac's full-mesh collision — a real cross-sim contact-semantics gap).
- Determinism: motrixsim stepping is deterministic across runs.

## Bugs found & fixed
1. **Stale PD feedback** — `write_data_to_sim` used cached `self._data.joint_pos` (one substep old);
   the mujoco backend reads live `mj_data.qpos`. Fixed to read fresh joint state each substep.
2. **NaN-divergence reset crash** — under early random RL actions the humanoid blows up; the reset path
   fed a NaN free-joint quaternion to `set_dof_pos` (which validates quaternions) → panic. Fixed:
   `_sanitize_dofs` scrubs NaN/Inf and renormalizes the quaternion on every `set_dof_pos`; high-threshold
   (200 rad/s) velocity clamp + panic-recovery in `MotrixSim.step` prevent/contain blow-ups.

## OPEN ISSUES (must resolve before trusting training)
1. **Control-fidelity discrepancy (unresolved).** A hand-written raw PD (compute torque in MJCF order,
   set `actuator_ctrls`, step) provably holds the G1 standing in both MuJoCo and MotrixSim (tilt 0.3°).
   `write_data_to_sim` — which computes the *same* PD — lets the robot tip, and a per-step comparison
   shows its `actuator_ctrls` diverging from the raw value (~8 Nm) once the robot moves. Yet every
   component verifies correct in isolation (reorder maps, gains, read order, gather index=identity,
   deterministic sim). This contradiction is unexplained — likely a subtle motrixsim binding interaction
   or a flaw in the microbenchmark harness. **This is the prime suspect for non-learning.**
2. **No clear RL learning.** Over ~110 iters / ~0.9M frames: episode_len 14→22 then flat; return
   declining (−396→−585); critic value collapses to ~0 (99.6% negative rewards). Inconclusive at this
   frame count (humanoid needs 10–100M) but not encouraging. CPU throughput makes a fair test slow.
3. **Throughput degrades over time** (~1000 → ~300 frames/s over ~100 iters) — a memory/state leak,
   likely in the per-substep `update()` (rebuilds the data dataclass + per-link velocity loop each call).
4. DR terms (`perturb_body_mass`, `_materials`, `motor_params`, `push_body`) not ported (registry skips).

## Recommended next steps
- Resolve Open Issue #1 first: instrument `write_data_to_sim` *inside the method* (print its own
  `actuator_ctrls` next to a raw computation using the method's own cached arrays) to find why the
  composite diverges when every part checks out. Consider replacing the data-layer PD with motrixsim's
  native position actuators (set actuator gains in the MJCF) to sidestep the Python PD path entirely.
- Fix the throughput leak (compute body velocities lazily / once per env-step; avoid `replace()` churn).
- Only then run a real training comparison (and consider porting DR).

## How to run
```bash
export CUDA_VISIBLE_DEVICES=""   # CPU only; GPU reserved for another task
# env: conda env_isaaclab (py3.11) — has motrixsim + torch + isaaclab; I added torchrl/jaxtyping/etc + `pip install -e . --no-deps`
python scripts/train_ppo.py task=G1/G1LocoFlat backend=motrixsim headless=true wandb.mode=disabled task.num_envs=256
```
(`max_iters` is ignored by train_ppo; control length via `total_frames`.)

## Files
Added: `backends/motrixsim/{motrixsim_sim,adapter,env,__init__}.py`, `NOTES.md`, this file.
Edited (added `motrixsim` wherever `mujoco` is handled): `__init__.py`, `helpers.py`, `envs/__init__.py`,
`assets/asset_cfg.py` (`.motrixsim()`), `registry.py` (`supported_backends`), `mdp/rewards/joint.py`.
All uncommitted on `main`.
