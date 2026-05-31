"""MotrixSim (Motphys) backend data layer.

Mirrors the IsaacLab-style ``Articulation.data`` API used by the MDP layer,
implemented on top of MotrixSim's ``SceneModel`` / ``SceneData`` (CPU, MJCF-native),
batched over ``num_envs``. This is the MotrixSim analog of ``backends/mujoco/mujoco.py``.

See ``NOTES.md`` in this directory for the full API mapping.
"""

import os
import tempfile
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Union

import mujoco
import numpy as np
import torch

import motrixsim as mx

try:
    from isaaclab.utils import string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils

from active_adaptation.utils.math import quat_rotate_inverse, quat_rotate

ArrayType = Union[np.ndarray, torch.Tensor]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


@dataclass
class MotrixArticulationCfg:
    mjcf_path: str
    init_state: Dict
    actuators: Dict
    body_names_simulation: Sequence[str]
    joint_names_simulation: Sequence[str]
    joint_symmetry_mapping: Dict = None
    spatial_symmetry_mapping: Dict = None


@dataclass
class MotrixTerrainCfg:
    mjcf_path: Optional[str] = None


@dataclass
class MotrixArticulationData:
    """Batched (num_envs, ...) articulation state, in *simulation* (Isaac) order."""

    default_joint_pos: ArrayType
    default_joint_vel: ArrayType
    default_root_state: ArrayType
    default_mass: ArrayType
    default_inertia: ArrayType

    joint_stiffness: ArrayType = None
    joint_damping: ArrayType = None
    joint_pos_limits: ArrayType = None

    body_link_pos_w: ArrayType = None
    body_link_quat_w: ArrayType = None  # wxyz
    body_com_pos_w: ArrayType = None

    joint_pos: ArrayType = None
    joint_pos_target: ArrayType = None
    joint_vel: ArrayType = None
    joint_vel_target: ArrayType = None
    joint_acc: ArrayType = None

    applied_torque: ArrayType = None
    projected_gravity_b: ArrayType = None

    body_com_vel_w: ArrayType = None  # (N, nbody, 6) [ang(3), lin(3)]
    heading_w: ArrayType = None

    # --- derived views (match backends/mujoco/mujoco.py:MJArticulationData) ---
    @property
    def body_lin_vel_w(self):
        return self.body_com_vel_w[..., 3:]

    @property
    def body_ang_vel_w(self):
        return self.body_com_vel_w[..., :3]

    @property
    def root_pos_w(self):
        return self.body_link_pos_w[..., 0, :]

    @property
    def root_quat_w(self):
        return self.body_link_quat_w[..., 0, :]

    @property
    def root_link_pos_w(self):
        return self.body_link_pos_w[:, 0, :]

    @property
    def root_com_pos_w(self):
        return self.body_com_pos_w[:, 0, :]

    @property
    def root_link_quat_w(self):
        return self.body_link_quat_w[:, 0, :]

    @property
    def root_link_pose_w(self):
        return torch.cat([self.body_link_pos_w[:, 0, :], self.body_link_quat_w[:, 0, :]], dim=-1)

    @property
    def root_com_lin_vel_w(self):
        return self.body_com_vel_w[:, 0, 3:]

    @property
    def root_com_ang_vel_w(self):
        return self.body_com_vel_w[:, 0, :3]

    @property
    def root_com_lin_vel_b(self):
        return quat_rotate_inverse(self.body_link_quat_w[:, 0, :], self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self):
        return quat_rotate_inverse(self.body_link_quat_w[:, 0, :], self.root_com_ang_vel_w)

    @property
    def root_state_w(self):
        return torch.cat([self.body_link_pos_w[:, 0, :], self.body_link_quat_w[:, 0, :]], dim=-1)

    # --- Isaac-style aliases (link-frame approximated by single-rigid-body link state) ---
    @property
    def body_quat_w(self):
        return self.body_link_quat_w

    @property
    def body_pos_w(self):
        return self.body_link_pos_w

    @property
    def body_link_vel_w(self):
        return self.body_com_vel_w

    @property
    def root_link_lin_vel_w(self):
        return self.body_com_vel_w[:, 0, 3:]

    @property
    def root_link_ang_vel_w(self):
        return self.body_com_vel_w[:, 0, :3]

    @property
    def root_link_ang_vel_b(self):
        return self.root_com_ang_vel_b

    @property
    def root_link_lin_vel_b(self):
        return self.root_com_lin_vel_b

    @property
    def root_link_state_w(self):
        return self.root_state_w


class MotrixArticulation:
    is_fixed_base = False

    def __init__(self, cfg: MotrixArticulationCfg):
        self.cfg = cfg
        # build a mujoco spec so we can (a) add a ground plane and (b) reuse mj gain setup
        self.spec = mujoco.MjSpec.from_file(cfg.mjcf_path)
        # FEET-ONLY collision (default). Two reasons, both required for Go2 to train:
        #  (1) MotrixSim's impulse solver panics ("LTL factorization NotPositiveDefinite")
        #      when the trunk/calf/thigh boxes lie flat on the floor during a fall (many
        #      coincident contacts -> singular contact matrix) -> NaN state -> poisons PPO.
        #      PhysX/isaac tolerate this; MotrixSim does not. Restricting collision to the
        #      foot point-contacts removes the degenerate configs (matches umi-on-legs).
        #  (2) We also name the kept collision geoms after their body so the contact sensor
        #      can map geom<->body; go2.xml's geoms are ALL unnamed, which is why the sensor
        #      built no pairs and feet_air_time/feet_sliding were silently DEAD.
        # Non-foot terms (undesired_contact / crash) then read 0, but with feet-only collision
        # those bodies can't touch the floor anyway; geometric fall_over handles termination.
        # Opt out (full-body collision) with MOTRIX_FULLBODY=1 for debugging.
        # FULL-BODY collision is the DEFAULT, matching isaac (the policy is trained
        # against the ground with all link collision geoms). Feet-ONLY collision (the
        # old default) made the Go2 an unstable inverted pendulum on 4 point-feet: with
        # the compliant explicit PD it tips and flips within ~1 s even at the default
        # pose, so no transferred policy can survive. Opt into feet-only (debug) with
        # MOTRIX_FEET_ONLY=1. The LTL-solver blow-ups that originally motivated feet-only
        # are cured here by stiff WELL-DAMPED contact (solref below) + Newton/multiccd
        # solver options set in MotrixScene (umi-on-LEGS recipe).
        feet_only = bool(os.environ.get("MOTRIX_FEET_ONLY"))
        # umi-on-LEGS contact: solref negative = direct (stiffness, damping). Damping
        # -5000 (vs MuJoCo's underdamped default) kills the contact chatter that makes
        # the impulse solver go NotPositiveDefinite.
        SOLREF = [-50000.0, -5000.0]
        SOLIMP = [0.99, 0.999, 0.0001, 0.5, 2.0]
        self.collision_geom_bodies: dict[str, str] = {}
        for body in self.spec.bodies:
            is_foot_body = ("foot" in body.name.lower() or "ankle" in body.name.lower())
            k = 0
            for g in body.geoms:
                if g.contype == 0 and g.conaffinity == 0:
                    continue  # visual-only geom
                if feet_only and not is_foot_body:
                    g.contype = 0
                    g.conaffinity = 0
                    continue
                # MotrixSim needs contype=1 AND conaffinity=1 (no MuJoCo bitmask OR-check)
                g.contype = 1
                g.conaffinity = 1
                g.solref = SOLREF
                g.solimp = SOLIMP
                if not g.name:
                    g.name = f"{body.name}_colgeom{k}"
                self.collision_geom_bodies[g.name] = body.name
                k += 1
        # Inject a motor (torque) actuator per hinge joint that lacks one. Some MJCFs
        # (e.g. go2.xml) declare <motor> only in <default>, yielding 0 real actuators;
        # this mirrors the mujoco backend so any robot is drivable.
        existing = {a.target for a in self.spec.actuators}
        for joint in self.spec.joints:
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE or joint.name in existing:
                continue
            act = self.spec.add_actuator(name=joint.name, target=joint.name)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.gear[0] = 1.0

    # ------------------------------------------------------------------
    def _initialize(self, model: "mx.SceneModel", data: "mx.SceneData", num_envs: int, device: str):
        self.model = model
        self.mtx_data = data
        self.num_envs = num_envs
        self.device = device
        self.body = model.get_body(self._find_root_body_name())

        # ---- name lists & ordering maps (motrixsim/MJCF order <-> Isaac order) ----
        self.joint_names_isaac = list(self.cfg.joint_names_simulation)
        self.body_names_isaac = list(self.cfg.body_names_simulation)
        self.joint_names_mtx = list(model.joint_names)
        self.link_names_mtx = list(model.link_names)
        self.actuator_target_mtx = [a.target_name for a in model.actuators]

        self._check_names("joint", self.joint_names_isaac, self.joint_names_mtx)
        self._check_names("body", self.body_names_isaac, self.link_names_mtx)

        self._jnt_mtx2isaac = [self.joint_names_mtx.index(n) for n in self.joint_names_isaac]
        self._body_mtx2isaac = [self.link_names_mtx.index(n) for n in self.body_names_isaac]
        # actuator k (motrixsim order) -> index into the Isaac-ordered torque vector
        self._act2isaac = [self.joint_names_isaac.index(t) for t in self.actuator_target_mtx]

        self._link_objs = [model.get_link(n) for n in self.link_names_mtx]
        # effort/torque limits per actuator: prefer the asset's effort_limit (works when
        # the MJCF has no ctrlrange, e.g. injected motors); else fall back to ctrl_range.
        eff_cfg = self.cfg.actuators.get("all", {}).get("effort_limit", None)
        if eff_cfg:
            eff_isaac = np.full(len(self.joint_names_isaac), 1.0e6, dtype=np.float32)
            ids, _, vals = string_utils.resolve_matching_names_values(eff_cfg, self.joint_names_isaac)
            eff_isaac[ids] = np.asarray(vals, dtype=np.float32)
            eff_act = eff_isaac[self._act2isaac]  # native/actuator order
            self.ctrl_limit_mtx = np.stack([-eff_act, eff_act], axis=1).astype(np.float32)
        else:
            self.ctrl_limit_mtx = np.asarray(
                [a.ctrl_range for a in model.actuators], dtype=np.float32
            )  # (nu, 2)

        # ---- defaults / gains in Isaac order ----
        nj = self.num_joints
        default_joint_pos = torch.zeros(nj)
        jids, jnames, jvals = string_utils.resolve_matching_names_values(
            self.cfg.init_state["joint_pos"], self.joint_names_isaac
        )
        default_joint_pos[jids] = torch.as_tensor(jvals, dtype=torch.float32)
        default_joint_vel = torch.zeros(nj)

        joint_stiffness = torch.zeros(nj)
        joint_damping = torch.zeros(nj)
        joint_armature = torch.zeros(nj)
        for _, actuator_cfg in self.cfg.actuators.items():
            for key, dst in (("stiffness", joint_stiffness), ("damping", joint_damping), ("armature", joint_armature)):
                spec = actuator_cfg.get(key, {".*": 0.0})
                ids, _, vals = string_utils.resolve_matching_names_values(spec, self.joint_names_isaac)
                dst[ids] = torch.as_tensor(vals, dtype=torch.float32)
        self._kp = joint_stiffness.numpy().astype(np.float32)
        self._kd = joint_damping.numpy().astype(np.float32)

        # joint pos limits in Isaac order, read from the compiled mujoco model (same MJCF)
        jl_isaac = np.zeros((nj, 2), dtype=np.float32)
        for i, jname in enumerate(self.joint_names_isaac):
            jl = self.model.get_joint(jname).range
            jl_isaac[i] = np.asarray(jl, dtype=np.float32)
        self._device_t = torch.device(device)

        N = self.num_envs
        self._data = MotrixArticulationData(
            default_joint_pos=default_joint_pos[None].repeat(N, 1).to(device),
            default_joint_vel=default_joint_vel[None].repeat(N, 1).to(device),
            default_root_state=torch.tensor(
                [[*self.cfg.init_state["pos"], *self.cfg.init_state["rot"], 0, 0, 0, 0, 0, 0]],
                dtype=torch.float32,
            ).repeat(N, 1).to(device),
            default_mass=torch.ones(N, self.num_bodies, device=device),
            default_inertia=torch.zeros(N, self.num_bodies, 9, device=device),
            joint_stiffness=torch.as_tensor(self._kp)[None].repeat(N, 1).to(device),
            joint_damping=torch.as_tensor(self._kd)[None].repeat(N, 1).to(device),
            joint_pos_limits=torch.as_tensor(jl_isaac)[None].repeat(N, 1, 1).to(device),
            applied_torque=torch.zeros(N, nj, device=device),
        )
        self._data.joint_pos_target = self._data.default_joint_pos.clone()
        self._data.joint_vel_target = self._data.default_joint_vel.clone()

        # external wrench buffers (body frame), in Isaac order
        self._external_force_b = torch.zeros(N, self.num_bodies, 3, device=device)
        self._external_torque_b = torch.zeros(N, self.num_bodies, 3, device=device)
        self.has_external_wrench = False

        self._prev_joint_vel = None
        self._prev_body_pos = None
        self._prev_body_quat = None
        # Gravity-compensation feedforward (umi-on-LEGS technique). Explicit PD has a
        # steady-state droop (kp*(target-q)=tau_gravity -> q sags), so Go2 can't even
        # hold its default pose (base sags 0.40->0.21) and RL can't bootstrap. The
        # feedforward removes the droop. Opt out with NO_GRAVITY_FF=1.
        self._gravity_ff = None
        if not os.environ.get("NO_GRAVITY_FF"):
            self._setup_gravity_ff()
        self.update(0.0)

    def _setup_gravity_ff(self):
        """Gravity-comp feedforward via umi-on-LEGS' droop-measure + scale-search.

        Phase 1: settle one env at the default pose with pure PD; the steady-state
        error ``droop = default - settled`` is the gravity-induced sag. The holding
        torque is ``kp*droop``. Phase 2: grid-search a scale (full comp overshoots)
        that minimises base-z drift. Result (native order) -> self._gravity_ff.
        """
        m = self.model
        mtxj = self.joint_names_mtx
        mfi = [self.joint_names_isaac.index(n) for n in mtxj]
        kp_n = self._kp[mfi]
        kd_n = self._kd[mfi]
        default_n = self._data.default_joint_pos[0].cpu().numpy()[mfi]
        cr = self.ctrl_limit_mtx
        nj = self.num_joints
        pos0 = np.asarray(self.cfg.init_state["pos"], dtype=np.float32)
        quat0 = wxyz_to_xyzw(np.asarray(self.cfg.init_state["rot"], dtype=np.float32))
        tmp = mx.SceneData(m, batch=(1,))

        def reset():
            dof = np.asarray(tmp.dof_pos).copy()
            dof[0, 0:3] = pos0
            dof[0, 3:7] = quat0
            dof[0, 7:7 + nj] = default_n
            tmp.set_dof_pos(dof.astype(np.float32), m)
            tmp.set_dof_vel(np.zeros_like(np.asarray(tmp.dof_vel), dtype=np.float32))
            m.forward_kinematic(tmp)

        def settle(offset, n):
            for _ in range(n):
                cur = np.asarray(tmp.dof_pos)[:, 7:7 + nj]
                cvel = np.asarray(tmp.dof_vel)[:, 6:6 + nj]
                tau = np.clip(kp_n * (default_n - cur) - kd_n * cvel + offset, cr[:, 0], cr[:, 1])
                tmp.actuator_ctrls = tau.astype(np.float32)
                m.step(tmp)

        # Go2 on point-feet is passively UNSTABLE with the soft PD (it tips during a
        # settle), so droop can't be measured directly. Settle with a STIFF PD instead:
        # it holds the default pose tightly (no sag, no tip) so the steady-state torque
        # IS the true gravity holding torque (robot genuinely standing, body load via
        # the feet included). Average it over the tail and apply with the normal kp.
        dbg = bool(os.environ.get("GRAVITY_FF_DEBUG"))
        kp_stiff = kp_n * 8.0
        kd_stiff = kd_n * 8.0
        reset()
        n_settle, n_avg = 1200, 300
        tau_acc = np.zeros(nj, dtype=np.float64)
        for step in range(n_settle + n_avg):
            cur = np.asarray(tmp.dof_pos)[:, 7:7 + nj]
            cvel = np.asarray(tmp.dof_vel)[:, 6:6 + nj]
            tau = np.clip(kp_stiff * (default_n - cur) - kd_stiff * cvel, cr[:, 0], cr[:, 1])
            tmp.actuator_ctrls = tau.astype(np.float32)
            m.step(tmp)
            if step >= n_settle:
                tau_acc += tau[0]
            if dbg and step in (50, 300, 800, n_settle + n_avg - 1):
                print(f"  [gravity_ff] step {step}: base_z={float(np.asarray(tmp.dof_pos)[0,2]):.3f} "
                      f"max|jdroop|={np.abs(default_n - cur[0]).max():.3f}rad")
        self._gravity_ff = (tau_acc / n_avg).astype(np.float32)
        if dbg:
            print(f"  [gravity_ff] max|ff|={np.abs(self._gravity_ff).max():.2f}Nm "
                  f"mean|ff|={np.abs(self._gravity_ff).mean():.2f}Nm")

    # ------------------------------------------------------------------
    def _find_root_body_name(self) -> str:
        # the first entry of body_names_simulation is the root link (pelvis/base/trunk)
        return list(self.cfg.body_names_simulation)[0]

    @staticmethod
    def _check_names(kind, isaac, mtx):
        if set(isaac) != set(mtx):
            warnings.warn(
                f"MotrixSim {kind} names mismatch:\n"
                f"  isaac-mtx: {set(isaac) - set(mtx)}\n  mtx-isaac: {set(mtx) - set(isaac)}",
                UserWarning,
            )

    @property
    def joint_names(self):
        return self.joint_names_isaac

    @property
    def body_names(self):
        return self.body_names_isaac

    @property
    def num_joints(self):
        return len(self.joint_names_isaac)

    @property
    def num_bodies(self):
        return len(self.body_names_isaac)

    @property
    def data(self):
        return self._data

    def find_bodies(self, name_keys, preserve_order: bool = False):
        return string_utils.resolve_matching_names(name_keys, self.body_names_isaac, preserve_order)

    def find_joints(self, name_keys, joint_subset=None, preserve_order: bool = False):
        if joint_subset is None:
            joint_subset = self.joint_names_isaac
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    # ---- env-id helpers ------------------------------------------------
    def _mask(self, env_ids):
        mask = np.zeros(self.num_envs, dtype=bool)
        if env_ids is None:
            mask[:] = True
        else:
            mask[np.asarray(env_ids.cpu() if torch.is_tensor(env_ids) else env_ids)] = True
        return mask

    @staticmethod
    def _quat_fd_angvel(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
        """World-frame angular velocity from a quaternion finite difference.

        q_prev, q_curr: (..., 4) wxyz; returns (..., 3). dq = q_curr * conj(q_prev).
        """
        w1, x1, y1, z1 = q_curr[..., 0], q_curr[..., 1], q_curr[..., 2], q_curr[..., 3]
        w2, x2, y2, z2 = q_prev[..., 0], -q_prev[..., 1], -q_prev[..., 2], -q_prev[..., 3]
        dw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        dx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        dy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        dz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        sign = np.where(dw < 0, -1.0, 1.0)[..., None]  # shortest path (double-cover)
        return (2.0 / dt) * sign * np.stack([dx, dy, dz], axis=-1)

    # ---- state read (motrixsim -> Isaac-ordered torch on device) -------
    # Reads base + joints by slicing dof_pos/dof_vel directly (qpos/qvel order),
    # and body velocities by finite-differencing the single batched get_link_poses
    # call. This mirrors umi-on-LEGS' motrix env and avoids the per-link velocity
    # loop (30x get_linear/angular_velocity calls) that throttled throughput.
    def update(self, dt: float):
        d = self.mtx_data
        N = self.num_envs
        nj = self.num_joints
        dof_pos = np.asarray(d.dof_pos)  # (N, nq) native qpos
        dof_vel = np.asarray(d.dof_vel)  # (N, nv) native qvel

        joint_pos = dof_pos[:, 7:7 + nj][:, self._jnt_mtx2isaac]
        joint_vel = dof_vel[:, 6:6 + nj][:, self._jnt_mtx2isaac]

        # base straight from the free joint: lin vel is world, ang vel is BODY frame
        root_quat_wxyz = xyzw_to_wxyz(dof_pos[:, 3:7])
        root_lin_vel_w = dof_vel[:, 0:3]
        root_ang_vel_b = dof_vel[:, 3:6]

        link_poses = np.asarray(self.model.get_link_poses(d)).reshape(N, self.model.num_links, 7)[:, self._body_mtx2isaac]
        body_link_pos_w = link_poses[..., :3]
        body_link_quat_w = xyzw_to_wxyz(link_poses[..., 3:7])

        if self._prev_body_pos is not None and dt:
            body_lin_vel_w = (body_link_pos_w - self._prev_body_pos) / dt
            body_ang_vel_w = self._quat_fd_angvel(self._prev_body_quat, body_link_quat_w, dt)
        else:
            body_lin_vel_w = np.zeros_like(body_link_pos_w)
            body_ang_vel_w = np.zeros_like(body_link_pos_w)
        self._prev_body_pos = body_link_pos_w
        self._prev_body_quat = body_link_quat_w

        # rotate body-frame root ang vel to world (numpy; avoid a torch round-trip per substep)
        qw = root_quat_wxyz[:, 0:1]
        qxyz = root_quat_wxyz[:, 1:4]
        t = 2.0 * np.cross(qxyz, root_ang_vel_b)
        root_ang_vel_w = root_ang_vel_b + qw * t + np.cross(qxyz, t)
        body_ang_vel_w[:, 0] = root_ang_vel_w
        body_lin_vel_w[:, 0] = root_lin_vel_w
        dev = self.device
        root_quat_t = torch.as_tensor(root_quat_wxyz, dtype=torch.float32, device=dev)
        body_com_vel_w = np.concatenate([body_ang_vel_w, body_lin_vel_w], axis=-1)  # [ang, lin]

        joint_pos_t = torch.as_tensor(joint_pos, dtype=torch.float32, device=dev)
        joint_vel_t = torch.as_tensor(joint_vel, dtype=torch.float32, device=dev)
        if self._prev_joint_vel is None or not dt:
            joint_acc_t = torch.zeros_like(joint_vel_t)
        else:
            joint_acc_t = (joint_vel_t - self._prev_joint_vel) / dt
        self._prev_joint_vel = joint_vel_t

        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3)
        projected_gravity_b = quat_rotate_inverse(root_quat_t, gravity_vec)
        w, x, y, z = root_quat_t.unbind(-1)
        heading_w = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        self._data = replace(
            self._data,
            joint_pos=joint_pos_t,
            joint_vel=joint_vel_t,
            joint_acc=joint_acc_t,
            joint_pos_target=self._data.joint_pos_target,
            joint_vel_target=self._data.joint_vel_target,
            body_link_pos_w=torch.as_tensor(body_link_pos_w, dtype=torch.float32, device=dev),
            body_link_quat_w=torch.as_tensor(body_link_quat_w, dtype=torch.float32, device=dev),
            body_com_pos_w=torch.as_tensor(body_link_pos_w, dtype=torch.float32, device=dev),
            body_com_vel_w=torch.as_tensor(body_com_vel_w, dtype=torch.float32, device=dev),
            projected_gravity_b=projected_gravity_b,
            heading_w=heading_w,
        )

    def reset(self, env_ids: ArrayType = None):
        # refresh cached data after the reset writes (mirrors the mujoco backend)
        self.update(0.0)

    # ---- control -------------------------------------------------------
    def set_joint_position_target(self, target: ArrayType, joint_ids: ArrayType = None):
        if joint_ids is None:
            self._data.joint_pos_target[:] = target
        else:
            self._data.joint_pos_target[:, joint_ids] = target

    def set_joint_velocity_target(self, target: ArrayType, joint_ids: ArrayType = None):
        if joint_ids is None:
            self._data.joint_vel_target[:] = target
        else:
            self._data.joint_vel_target[:, joint_ids] = target

    def write_data_to_sim(self):
        # PD-as-torque computed entirely in native (MJCF joint) order and gathered to
        # actuators by joint index — the formulation verified to hold the robot. Read
        # FRESH joint state each substep (the cached self._data.joint_pos lags the PD
        # feedback by one substep and destabilizes the inverted-pendulum humanoid).
        N = self.num_envs
        if not hasattr(self, "_pd_kp_mtx"):
            mtxj = self.joint_names_mtx
            # isaac-ordered target -> native order; native joint -> isaac (for applied_torque)
            self._mtx_from_isaac = [self.joint_names_isaac.index(n) for n in mtxj]
            # actuator k -> index of its target joint in native order
            self._act_jointidx = [mtxj.index(t) for t in self.actuator_target_mtx]
            # explicit PD is SOFTER + underdamped vs isaac's implicit PD at the same
            # gains (isaac's implicit drive tracks the target far more stiffly per step).
            # KP_SCALE/KD_SCALE calibrate the explicit gains up to reproduce isaac's
            # effective stiffness/damping so an isaac-trained policy transfers.
            self._pd_kp_mtx = self._kp[self._mtx_from_isaac] * float(os.environ.get("KP_SCALE", "1.0"))
            self._pd_kd_mtx = self._kd[self._mtx_from_isaac] * float(os.environ.get("KD_SCALE", "1.0"))

        # current joint state by slicing dof_pos/dof_vel (native qpos/qvel order)
        nj = self.num_joints
        jp_mtx = np.asarray(self.mtx_data.dof_pos)[:, 7:7 + nj]
        jv_mtx = np.asarray(self.mtx_data.dof_vel)[:, 6:6 + nj]
        tgt_mtx = self._data.joint_pos_target.detach().cpu().numpy()[:, self._mtx_from_isaac]
        vtgt_mtx = self._data.joint_vel_target.detach().cpu().numpy()[:, self._mtx_from_isaac]
        tau_mtx = self._pd_kp_mtx * (tgt_mtx - jp_mtx) + self._pd_kd_mtx * (vtgt_mtx - jv_mtx)
        if getattr(self, "_gravity_ff", None) is not None:
            tau_mtx = tau_mtx + self._gravity_ff  # gravity-compensation feedforward (native)
        tau_mtx = np.nan_to_num(tau_mtx, nan=0.0, posinf=0.0, neginf=0.0)
        # clip to the effort limits in NATIVE order, so applied_torque (exposed to the
        # reward terms) reflects what is ACTUALLY applied (not the unclipped PD demand,
        # which spiked to +-85 Nm during a blow-up and wrecked the torque/energy rewards).
        if not hasattr(self, "_eff_lo_native"):
            self._eff_lo_native = np.empty(self.num_joints, np.float32)
            self._eff_hi_native = np.empty(self.num_joints, np.float32)
            self._eff_lo_native[self._act_jointidx] = self.ctrl_limit_mtx[:, 0]
            self._eff_hi_native[self._act_jointidx] = self.ctrl_limit_mtx[:, 1]
        tau_mtx = np.clip(tau_mtx, self._eff_lo_native, self._eff_hi_native)
        self._data.applied_torque = torch.as_tensor(
            tau_mtx[:, self._jnt_mtx2isaac], dtype=torch.float32, device=self.device
        )
        ctrl = tau_mtx[:, self._act_jointidx]  # (N, nu) actuator order (already clipped)
        self.mtx_data.actuator_ctrls = ctrl.astype(np.float32)

        if self.has_external_wrench:
            # apply body-frame external force on the root as a world-frame force on base link
            from active_adaptation.utils.math import quat_rotate
            root_quat = self._data.root_link_quat_w.unsqueeze(1)
            force_w = quat_rotate(root_quat, self._external_force_b).cpu().numpy()
            for bi, link in enumerate(self._link_objs):
                pass  # per-body external force is applied in MotrixScene via add_external_force

    # ---- state write ---------------------------------------------------
    @staticmethod
    def _sanitize_dofs(dof: np.ndarray, dvel: np.ndarray) -> tuple:
        """Scrub NaN/Inf and renormalize the free-joint quaternion (dof[:,3:7], xyzw).

        A diverged env (blown up to NaN) would otherwise crash motrixsim's
        ``set_dof_pos``, which validates that dof[3:7] is a normalized quaternion.
        Such envs are reset to an upright identity orientation and will terminate.
        """
        dof = np.nan_to_num(dof, nan=0.0, posinf=0.0, neginf=0.0)
        dvel = np.nan_to_num(dvel, nan=0.0, posinf=0.0, neginf=0.0)
        q = dof[:, 3:7]
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (norm[:, 0] < 1e-6)
        q = q / np.where(norm < 1e-6, 1.0, norm)
        q[bad] = np.array([0.0, 0.0, 0.0, 1.0], dtype=dof.dtype)  # identity xyzw
        dof[:, 3:7] = q
        return dof, dvel

    def write_root_state_to_sim(self, root_state: ArrayType, env_ids: ArrayType = None):
        mask = self._mask(env_ids)
        root_state = root_state.detach().cpu().numpy() if torch.is_tensor(root_state) else np.asarray(root_state)
        dof = np.asarray(self.mtx_data.dof_pos).copy()
        dvel = np.asarray(self.mtx_data.dof_vel).copy()
        idx = np.nonzero(mask)[0]
        dof[idx, 0:3] = root_state[:, 0:3]
        dof[idx, 3:7] = wxyz_to_xyzw(root_state[:, 3:7])
        # default joints (Isaac order) -> motrixsim order
        default_jpos = self._data.default_joint_pos.detach().cpu().numpy()
        jpos_mtx = np.zeros((len(idx), self.num_joints), dtype=np.float32)
        jpos_mtx[:, self._jnt_mtx2isaac] = default_jpos[idx]
        dof[np.ix_(idx, np.arange(7, 7 + self.num_joints))] = jpos_mtx
        if root_state.shape[1] >= 13:
            dvel[idx, 0:3] = root_state[:, 7:10]
            dvel[idx, 3:6] = root_state[:, 10:13]
        else:
            dvel[idx, 0:6] = 0.0
        dvel[np.ix_(idx, np.arange(6, 6 + self.num_joints))] = 0.0
        dof, dvel = self._sanitize_dofs(dof, dvel)
        self.mtx_data.set_dof_pos(dof.astype(np.float32), self.model)
        self.mtx_data.set_dof_vel(dvel.astype(np.float32))

    def write_joint_state_to_sim(self, joint_pos, joint_vel, joint_ids, env_ids: ArrayType = None):
        mask = self._mask(env_ids)
        idx = np.nonzero(mask)[0]
        dof = np.asarray(self.mtx_data.dof_pos).copy()
        dvel = np.asarray(self.mtx_data.dof_vel).copy()
        # current joints (motrixsim order) for these envs
        if joint_pos is not None:
            jp = joint_pos.detach().cpu().numpy() if torch.is_tensor(joint_pos) else np.asarray(joint_pos)
            cur = dof[np.ix_(idx, np.arange(7, 7 + self.num_joints))]
            cur_isaac = cur[:, [self.joint_names_mtx.index(n) for n in self.joint_names_isaac]]
            if joint_ids is None or isinstance(joint_ids, slice):
                cur_isaac[:, :] = jp
            else:
                cur_isaac[:, joint_ids] = jp
            jpos_mtx = np.zeros_like(cur)
            jpos_mtx[:, self._jnt_mtx2isaac] = cur_isaac
            dof[np.ix_(idx, np.arange(7, 7 + self.num_joints))] = jpos_mtx
        if joint_vel is not None:
            jv = joint_vel.detach().cpu().numpy() if torch.is_tensor(joint_vel) else np.asarray(joint_vel)
            cur = dvel[np.ix_(idx, np.arange(6, 6 + self.num_joints))]
            cur_isaac = cur[:, [self.joint_names_mtx.index(n) for n in self.joint_names_isaac]]
            if joint_ids is None or isinstance(joint_ids, slice):
                cur_isaac[:, :] = jv
            else:
                cur_isaac[:, joint_ids] = jv
            jvel_mtx = np.zeros_like(cur)
            jvel_mtx[:, self._jnt_mtx2isaac] = cur_isaac
            dvel[np.ix_(idx, np.arange(6, 6 + self.num_joints))] = jvel_mtx
        dof, dvel = self._sanitize_dofs(dof, dvel)
        self.mtx_data.set_dof_pos(dof.astype(np.float32), self.model)
        self.mtx_data.set_dof_vel(dvel.astype(np.float32))


@dataclass
class MotrixContactData:
    net_forces_w: ArrayType = None
    last_air_time: ArrayType = None
    current_air_time: ArrayType = None
    last_contact_time: ArrayType = None
    current_contact_time: ArrayType = None


class MotrixContactSensor:
    """Foot-ground contact via geom-pair ``is_colliding`` (feet-only collision model).

    Produces a per-body boolean contact -> contact/air-time bookkeeping, matching
    backends/mujoco/mujoco.py:MjContactSensor (which only uses force>thr as a boolean).
    Non-foot bodies have no floor contact pairs -> always 0 (matches the mujoco backend).
    """

    def __init__(self, articulation: MotrixArticulation):
        self.articulation = articulation

    def _initialize(self, model, mtx_data, dt: float):
        self.model = model
        self.mtx_data = mtx_data
        self.dt = dt
        self.device = self.articulation.device
        N = self.articulation.num_envs
        nb = self.articulation.num_bodies

        # map EVERY collision geom -> its Isaac body index, via the geom<->body names the
        # articulation recorded when it named them. Each (geom, floor) pair is queried and
        # OR-aggregated per body in update(), so feet (air_time/sliding), calf/thigh/head
        # (undesired_contact) and base (crash) all get real contact signals.
        geom_names = {n: i for i, n in enumerate(model.geom_names) if n is not None}
        self.floor_gid = geom_names.get("floor")
        body_names_isaac = self.articulation.body_names_isaac
        pair_list, pair_body = [], []
        if self.floor_gid is not None:
            for gname, body in self.articulation.collision_geom_bodies.items():
                gid = geom_names.get(gname)
                if gid is None or body not in body_names_isaac:
                    continue
                pair_list.append([gid, self.floor_gid])
                pair_body.append(body_names_isaac.index(body))
        self._pairs = np.asarray(pair_list, dtype=np.uint32) if pair_list else None
        self._pair_body = np.asarray(pair_body, dtype=np.int64) if pair_body else None

        z = lambda *s: torch.zeros(*s, device=self.device)
        self._data = MotrixContactData(
            net_forces_w=z(N, nb, 3),
            last_air_time=z(N, nb),
            current_air_time=z(N, nb),
            last_contact_time=z(N, nb),
            current_contact_time=z(N, nb),
        )
        self._cq = model.get_contact_query(mtx_data)

    def find_bodies(self, name_keys, preserve_order: bool = False):
        return self.articulation.find_bodies(name_keys, preserve_order)

    @property
    def data(self):
        return self._data

    def compute_first_contact(self, dt, abs_tol: float = 1.0e-8):
        currently_in_contact = self._data.current_contact_time > 0.0
        less_than_dt = self._data.current_contact_time < (dt + abs_tol)
        return (currently_in_contact & less_than_dt).float()

    def compute_first_air(self, dt, abs_tol: float = 1.0e-8):
        currently_detached = self._data.current_air_time > 0.0
        less_than_dt = self._data.current_air_time < (dt + abs_tol)
        return (currently_detached & less_than_dt).float()

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        for f in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
            getattr(self._data, f)[env_ids] = 0.0

    def update(self, dt: float):
        N = self.articulation.num_envs
        nb = self.articulation.num_bodies
        is_contact_body = torch.zeros(N, nb, dtype=torch.bool, device=self.device)
        if self._pairs is not None:
            self._cq = self.model.get_contact_query(self.mtx_data)
            colliding = np.asarray(self._cq.is_colliding(self._pairs)).reshape(N, -1)  # (N, P)
            colliding_t = torch.as_tensor(colliding, device=self.device)
            for p in range(self._pair_body.shape[0]):
                bi = int(self._pair_body[p])
                is_contact_body[:, bi] |= colliding_t[:, p]
        # pseudo net force along +z so downstream norm>0.1 detection works
        net = torch.zeros(N, nb, 3, device=self.device)
        net[..., 2] = is_contact_body.float() * 1.0
        self._data.net_forces_w = net

        elapsed = torch.as_tensor(dt, device=self.device)
        is_contact = is_contact_body
        is_first_contact = (self._data.current_air_time > 0) * is_contact
        is_first_detached = (self._data.current_contact_time > 0) * ~is_contact
        self._data.last_air_time = torch.where(
            is_first_contact, self._data.current_air_time + elapsed, self._data.last_air_time
        )
        self._data.current_air_time = torch.where(
            ~is_contact, self._data.current_air_time + elapsed, torch.zeros_like(self._data.current_air_time)
        )
        self._data.last_contact_time = torch.where(
            is_first_detached, self._data.current_contact_time + elapsed, self._data.last_contact_time
        )
        self._data.current_contact_time = torch.where(
            is_contact, self._data.current_contact_time + elapsed, torch.zeros_like(self._data.current_contact_time)
        )


class MotrixScene:
    def __init__(self, cfg, num_envs: int, device: str, physics_dt: float, step_dt: float = 0.02):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        # step_n mode (umi-on-legs): step all substeps natively in one call with the torque
        # held, instead of env_base's per-substep Python loop. round() fixes the int(0.02/0.002)=9 bug.
        # This is the default: it fixes the per-substep marginal PD instability (robot now
        # trains to 100% success vs ~23% before) AND is ~40x faster. Opt out with NO_STEPN=1.
        self.step_dt = step_dt
        self.substeps = max(1, round(step_dt / physics_dt))
        self.use_stepn = not bool(os.environ.get("NO_STEPN"))
        self.articulations: Dict[str, MotrixArticulation] = {}
        self.sensors: Dict[str, MotrixContactSensor] = {}

        # build combined spec: robot(s) + ground plane named "floor".
        # gather attributes from both the instance and its class hierarchy
        # (SceneCfg may declare `robot` as a class attribute).
        items = {}
        for klass in reversed(type(cfg).__mro__):
            items.update({k: v for k, v in vars(klass).items() if not k.startswith("__")})
        items.update(vars(cfg))
        robot_cfg = None
        for name, c in items.items():
            if isinstance(c, MotrixArticulationCfg):
                robot_cfg = c
                self.articulations[name] = MotrixArticulation(c)
        assert robot_cfg is not None, "MotrixScene requires a robot MotrixArticulationCfg"

        spec = self.articulations["robot"].spec
        if spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
            pass
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.name = "floor"
        g.size = [0.0, 0.0, 0.05]
        g.friction = [1.0, 0.1, 0.1]
        # STIFF contact (umi-on-LEGS values). Without an explicit solref the plane
        # uses MuJoCo's soft default and the Go2 feet sink ~0.7 m THROUGH the floor
        # (the robot "stands" at base_z=-0.3), which no policy can survive. Negative
        # solref = direct (stiffness, damping) = (50000 N/m, 200). priority=1 so the
        # foot<->floor contact adopts the floor's stiff params over the foot default.
        g.solref = [-50000.0, -5000.0]
        g.solimp = [0.99, 0.999, 0.0001, 0.5, 2.0]
        g.priority = 1
        g.condim = 3
        # solver options (umi-on-LEGS recipe): Newton + multiccd makes full-body
        # ground contact numerically stable on MotrixSim's impulse solver.
        try:
            spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
            spec.option.iterations = int(os.environ.get("MOTRIX_ITERS", "50"))
            spec.option.ls_iterations = 20
            spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            spec.option.impratio = 1.0
            if not os.environ.get("NO_MULTICCD"):
                spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        except Exception as _e:
            warnings.warn(f"could not set motrix solver options: {_e}")
        spec.compile()
        # write the combined scene next to the original MJCF so relative mesh
        # paths (e.g. meshes/*.STL) resolve, then load and clean up.
        mjcf_dir = os.path.dirname(os.path.abspath(robot_cfg.mjcf_path))
        fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=mjcf_dir, prefix="_motrix_scene_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(spec.to_xml())
            self.model = mx.load_model(tmp_path)
        finally:
            os.remove(tmp_path)
        self.model.options.timestep = physics_dt
        self.model.options.max_iterations = int(os.environ.get("MOTRIX_ITERS", "50"))
        self.mtx_data = mx.SceneData(self.model, batch=(num_envs,))
        self.model.forward_kinematic(self.mtx_data)

        for art in self.articulations.values():
            art._initialize(self.model, self.mtx_data, num_envs, device)
        # contact sensor (named to match asset ContactSensorCfg.name == "contact_forces")
        sensor = MotrixContactSensor(self.articulations["robot"])
        sensor._initialize(self.model, self.mtx_data, physics_dt)
        self.sensors["contact_forces"] = sensor

        self.env_origins = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids):
        for art in self.articulations.values():
            art.reset(env_ids)
        for s in self.sensors.values():
            s.reset(env_ids)

    def update(self, dt: float):
        for art in self.articulations.values():
            art.update(dt)
        for s in self.sensors.values():
            s.update(dt)

    def write_data_to_sim(self):
        for art in self.articulations.values():
            art.write_data_to_sim()

    def __getitem__(self, key):
        return self.articulations.get(key) or self.sensors.get(key)


class MotrixSim:
    device = "cpu"

    # safety bounds to keep the implicit solver well-conditioned on CPU
    VEL_CLAMP = 50.0

    def __init__(self, scene: MotrixScene):
        self.scene = scene
        self.model = scene.model
        self.mtx_data = scene.mtx_data

    def get_physics_dt(self):
        # In step_n mode, report step_dt so env_base runs ONE step per control step
        # (torque computed once); the real substeps happen inside step() via step_n.
        if self.scene.use_stepn:
            return float(self.scene.step_dt)
        return float(self.model.options.timestep)

    def has_gui(self):
        return False

    # clamp only true blow-ups; high enough not to touch normal humanoid motion
    VEL_CLAMP = 200.0

    def step(self, render: bool = False):
        # Scrub NaN/Inf and clamp runaway velocities before the solve so a diverging
        # env cannot blow up to NaN (which then crashes set_dof_pos at reset).
        dv = np.asarray(self.mtx_data.dof_vel)
        if (not np.isfinite(dv).all()) or (np.abs(np.nan_to_num(dv)).max() > self.VEL_CLAMP):
            dv = np.clip(np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0), -self.VEL_CLAMP, self.VEL_CLAMP)
            self.mtx_data.set_dof_vel(dv.astype(np.float32))
        try:
            if self.scene.use_stepn:
                # umi-on-legs pattern: step one substep with the env_base-computed torque,
                # recompute the PD once, then step_n the remaining substeps with it held.
                self.model.step(self.mtx_data)
                if self.scene.substeps > 1:
                    self.scene.write_data_to_sim()
                    self.model.step_n(self.mtx_data, self.scene.substeps - 1)
            else:
                self.model.step(self.mtx_data)
        except BaseException as e:  # pyo3 PanicException (e.g. NotPositiveDefinite)
            # in-place recovery: keep (sanitized) positions, zero velocities, drop ctrl
            warnings.warn(f"motrixsim step failed ({type(e).__name__}); recovering by zeroing velocities.")
            dp = np.asarray(self.mtx_data.dof_pos)
            dz = np.zeros_like(np.asarray(self.mtx_data.dof_vel))
            dp, dz = MotrixArticulation._sanitize_dofs(dp, dz)
            self.mtx_data.set_dof_pos(dp.astype(np.float32), self.model)
            self.mtx_data.set_dof_vel(dz.astype(np.float32))
            self.mtx_data.actuator_ctrls = np.zeros((self.scene.num_envs, self.model.num_actuators), dtype=np.float32)
            try:
                self.model.step(self.mtx_data)
            except BaseException:
                pass
