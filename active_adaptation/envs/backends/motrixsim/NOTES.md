# MotrixSim backend — API mapping notes

MotrixSim (Motphys) 0.6.1, CPU (Rust), MJCF-native. Installed in conda env `env_isaaclab` (py3.11).
Docs/examples cloned to /tmp/motrixsim-docs (legged_gym/, examples/). Reference RL impl: legged_gym.

## Core API (model/data split, MuJoCo-like)
- `m = mx.load_model(path)` / `mx.load_mjcf_str(xml)` -> `SceneModel`
- `d = mx.SceneData(m, batch=(N,))` -> batched state; fields `d.dof_pos (N,ndofpos)`, `d.dof_vel (N,ndofvel)`, `d.actuator_ctrls (N,nu)`
- `m.step(d)` (== `mx.step(m,d)`) steps ALL envs
- `d.set_dof_pos(arr, m)`  (NOTE arg order: array THEN model)
- `d.set_dof_vel(arr)`     (NOTE: NO model arg — inconsistent with set_dof_pos)
- `d.actuator_ctrls = arr` (settable property; read-only for dof_pos/dof_vel)
- `d.reset(m)`; subset via BOOL MASK only: `d[mask].set_dof_pos(...)` (int-array index NOT supported)

## Layout
- `dof_pos` (36 for G1) = [base_pos(3), base_quat(4, **xyzw**), joints(29 in MJCF order)]
- `dof_vel` (35) = [base_linvel(3), base_angvel(3), jointvel(29)]
- **Quaternion is XYZW**; codebase uses WXYZ -> convert: wxyz = xyzw[...,[3,0,1,2]]; xyzw = wxyz[...,[1,2,3,0]]

## Robot / bodies
- `b = m.get_body("pelvis")`; `b.get_joint_dof_pos(d)`/`get_joint_dof_vel(d)` -> (N,29) actuated joints (MJCF order)
- `b.base_link.get_pose(d)` -> (N,7) [pos, quat xyzw]; `.get_linear_velocity(d)`/`.get_angular_velocity(d)` -> (N,3) world
- `m.get_link_poses(d)` -> (N, num_links, 7) batched; per-link velocity only (no batched link-vel API)
- `Link.add_external_force/torque`, `Link.set_mass_override` (for push / mass DR)

## Ordering (CRITICAL)
- motrixsim joint/link order == MJCF declaration order (legs, then waist, then arms).
- codebase MDP expects `joint_names_simulation` / `body_names_simulation` = Isaac BFS order (interleaved L/R).
- Must reorder both ways, like mujoco backend's _jnt_mjc2isaac maps.

## Actuators / control
- repo g1.xml has 29 `<motor>` actuators (torque), ctrlrange = effort limit (±88/139/50 ...).
- PD-as-torque in Python: `ctrl = kp*(target - q) - kd*qd`, clip to ctrlrange (same as mujoco backend & legged_gym).

## Contacts (key correctness divergence)
- repo g1.xml is FEET-ONLY collision: geoms default contype=0/conaffinity=0; only 14 explicit `<pair>`
  (foot{1..7}_collision x `floor`, friction 0.8) collide. Non-foot bodies CANNOT contact ground
  (unlike Isaac full-mesh) -> `crash` termination (body-ground contact) effectively never fires here;
  `fall_over` (orientation) does. Matches the mujoco sim2sim backend.
- Need a ground geom named `floor`: build combined scene = g1.xml + plane geom("floor") via mujoco.MjSpec
  -> spec.to_xml() -> mx.load_model(tmp). (repo plane.xml provides `floor`.)
- Contact query is BATCHED: `cq = m.get_contact_query(d); cq.is_colliding(pairs (P,2) uint32) -> (N,P) bool`.
- No per-body net contact FORCE in high-level API (only is_colliding + low.get_contacts depth/normal).
  -> derive boolean foot contact -> contact-time bookkeeping (mujoco backend only uses force>0.1 as boolean anyway).

## GPU
- Pure CPU. Run everything with CUDA_VISIBLE_DEVICES="" (other task owns the GPU).
