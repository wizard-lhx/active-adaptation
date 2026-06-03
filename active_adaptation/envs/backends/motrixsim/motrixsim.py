"""MotrixSim (Motphys) backend data layer.

Mirrors the IsaacLab-style ``Articulation.data`` API used by the MDP layer,
implemented on top of MotrixSim's ``SceneModel`` / ``SceneData`` (CPU, MJCF-native),
batched over ``num_envs``. This is the MotrixSim analog of ``backends/mujoco/mujoco.py``.
"""

import os
import tempfile
import warnings
from dataclasses import dataclass, replace
from typing import Dict, Sequence, Union

import mujoco
import numpy as np
import torch

import motrixsim as mx

try:
    from isaaclab.utils import string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils

from active_adaptation.utils.math import quat_rotate_inverse

ArrayType = Union[np.ndarray, torch.Tensor]

# Stiff, well-damped contact (negative solref = direct stiffness/damping). The strong
# damping -5000 (vs MuJoCo's underdamped default) is what stops MotrixSim's impulse solver
# from going "LTL NotPositiveDefinite" under full-body ground contact. Applied to every
# collision geom and to the floor.
_CONTACT_SOLREF = [-50000.0, -5000.0]
_CONTACT_SOLIMP = [0.99, 0.999, 0.0001, 0.5, 2.0]


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
    """One robot: MJCF/scene setup, native<->Isaac translation, and the read/write
    paths the MDP and ``_EnvBase`` step loop use.

    All ``data`` fields are batched ``(num_envs, ...)`` and in Isaac order. Physics
    state is read by slicing ``dof_pos``/``dof_vel`` (native MJCF order); the
    permutation maps built in ``_initialize`` bridge the two orderings.
    """

    is_fixed_base = False

    def __init__(self, cfg: MotrixArticulationCfg):
        self.cfg = cfg
        # Edit the MJCF as a mujoco.MjSpec before MotrixSim loads it (no post-load edits
        # are possible): configure collision, then ensure every joint has an actuator.
        self.spec = mujoco.MjSpec.from_file(cfg.mjcf_path)

        # Collision model. Full-body is the default (matches Isaac, which collides all
        # link geoms against the ground); MOTRIX_FEET_ONLY=1 restricts collision to
        # foot/ankle bodies for debugging. Two MotrixSim-specific requirements:
        #   - both contype AND conaffinity must be 1 (no MuJoCo bitmask OR-check), and
        #   - every kept collision geom must be named, so the contact sensor can map it
        #     back to its body (go2.xml's geoms are unnamed).
        feet_only = bool(os.environ.get("MOTRIX_FEET_ONLY"))
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
                g.solref = _CONTACT_SOLREF
                g.solimp = _CONTACT_SOLIMP
                if not g.name:
                    g.name = f"{body.name}_colgeom{k}"
                self.collision_geom_bodies[g.name] = body.name
                k += 1
        # Resolve PD gains and configure native position-servo actuators (the engine
        # computes the PD implicitly each substep).
        self.joint_names_isaac = list(cfg.joint_names_simulation)
        self._kp, self._kd = self._resolve_gains()
        self._setup_actuators()

    def _resolve_gains(self):
        """Per-joint PD gains (kp, kd) from cfg, as Isaac-order numpy arrays."""
        nj = len(self.joint_names_isaac)
        kp = np.zeros(nj, dtype=np.float32)
        kd = np.zeros(nj, dtype=np.float32)
        for _, actuator_cfg in self.cfg.actuators.items():
            for key, dst in (("stiffness", kp), ("damping", kd)):
                spec = actuator_cfg.get(key, {".*": 0.0})
                ids, _, vals = string_utils.resolve_matching_names_values(spec, self.joint_names_isaac)
                dst[ids] = np.asarray(vals, dtype=np.float32)
        return kp, kd

    def _setup_actuators(self):
        """Configure a native MuJoCo position servo on every hinge joint.

        force = kp*(target-q) - kd*qd, via gainprm/biasprm, which MotrixSim integrates
        implicitly each substep. Joints with no actuator (e.g. go2.xml declares motors
        only in <default>) get one added, mirroring the mujoco backend.
        """
        kp_by_name = dict(zip(self.joint_names_isaac, self._kp))
        kd_by_name = dict(zip(self.joint_names_isaac, self._kd))
        by_target = {a.target: a for a in self.spec.actuators}
        for joint in self.spec.joints:
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            act = by_target.get(joint.name)
            if act is None:
                act = self.spec.add_actuator(name=joint.name, target=joint.name)
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gear[0] = 1.0
            kp = float(kp_by_name.get(joint.name, 0.0))
            kd = float(kd_by_name.get(joint.name, 0.0))
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm[0] = kp
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.biasprm[0] = 0.0
            act.biasprm[1] = -kp
            act.biasprm[2] = -kd

    # ------------------------------------------------------------------
    def _initialize(self, model: "mx.SceneModel", data: "mx.SceneData", num_envs: int, device: str):
        self.model = model
        self.mtx_data = data
        self.num_envs = num_envs
        self.device = device

        # ---- name lists & ordering maps (motrixsim/MJCF order <-> Isaac order) ----
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

        # ---- control maps: Isaac<->native joint perms + per-joint effort bounds ----
        # _jnt_mtx2isaac (above) goes native->Isaac; _mtx_from_isaac is its inverse.
        self._mtx_from_isaac = [self.joint_names_isaac.index(n) for n in self.joint_names_mtx]
        self._act_jointidx = [self.joint_names_mtx.index(t) for t in self.actuator_target_mtx]
        self._kp_native = self._kp[self._mtx_from_isaac]
        self._kd_native = self._kd[self._mtx_from_isaac]
        self._eff_lo_native = np.empty(self.num_joints, np.float32)
        self._eff_hi_native = np.empty(self.num_joints, np.float32)
        self._eff_lo_native[self._act_jointidx] = self.ctrl_limit_mtx[:, 0]
        self._eff_hi_native[self._act_jointidx] = self.ctrl_limit_mtx[:, 1]

        # ---- defaults in Isaac order (gains were resolved in __init__) ----
        nj = self.num_joints
        default_joint_pos = torch.zeros(nj)
        jids, _, jvals = string_utils.resolve_matching_names_values(
            self.cfg.init_state["joint_pos"], self.joint_names_isaac
        )
        default_joint_pos[jids] = torch.as_tensor(jvals, dtype=torch.float32)
        default_joint_vel = torch.zeros(nj)

        # joint pos limits in Isaac order, read from the compiled mujoco model (same MJCF)
        jl_isaac = np.zeros((nj, 2), dtype=np.float32)
        for i, jname in enumerate(self.joint_names_isaac):
            jl = self.model.get_joint(jname).range
            jl_isaac[i] = np.asarray(jl, dtype=np.float32)

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

        # External-wrench buffers (Isaac order). Kept for SceneAdapter parity (the
        # adapter zeroes them each substep); applying them to the sim is not implemented.
        self._external_force_b = torch.zeros(N, self.num_bodies, 3, device=device)
        self._external_torque_b = torch.zeros(N, self.num_bodies, 3, device=device)
        self.has_external_wrench = False

        self._prev_joint_vel = None
        self._prev_body_pos = None
        self._prev_body_quat = None
        self.update(0.0)

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

    def update(self, dt: float):
        """Read native MotrixSim state into the Isaac-ordered ``data`` (per substep).

        Base and joints are sliced from ``dof_pos``/``dof_vel``; per-body velocities come
        from finite-differencing one batched ``get_link_poses`` call (the per-link
        velocity API is too slow to loop). The root velocity rows are then overwritten
        with the exact free-joint velocities.
        """
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

        # replace() preserves the fields not listed (e.g. joint_pos_target, defaults).
        self._data = replace(
            self._data,
            joint_pos=joint_pos_t,
            joint_vel=joint_vel_t,
            joint_acc=joint_acc_t,
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
        """Command the joint-position target to the native servo; expose ``applied_torque``.

        MotrixSim integrates the PD (kp*(target-q) - kd*qd) implicitly each substep, so we
        only write the position target. ``applied_torque`` is reported analytically from
        FRESH joint state (clipped to the effort limits) for the reward terms.
        """
        # fresh joint state and target, native (qpos/qvel) order
        nj = self.num_joints
        jp_mtx = np.asarray(self.mtx_data.dof_pos)[:, 7:7 + nj]
        jv_mtx = np.asarray(self.mtx_data.dof_vel)[:, 6:6 + nj]
        tgt_mtx = self._data.joint_pos_target.detach().cpu().numpy()[:, self._mtx_from_isaac]

        # Command the position target. The array MUST be C-contiguous: MotrixSim reads
        # actuator_ctrls by raw buffer and ignores numpy strides, so a fancy-indexed
        # (non-contiguous) array scrambles per-env commands -> N>1 collapse (invisible at N=1).
        self.mtx_data.actuator_ctrls = np.ascontiguousarray(
            tgt_mtx[:, self._act_jointidx], dtype=np.float32
        )
        # report the servo torque (clipped to effort limits) for the reward terms
        tau_mtx = self._kp_native * (tgt_mtx - jp_mtx) - self._kd_native * jv_mtx
        tau_mtx = np.nan_to_num(tau_mtx, nan=0.0, posinf=0.0, neginf=0.0)
        tau_mtx = np.clip(tau_mtx, self._eff_lo_native, self._eff_hi_native)
        self._data.applied_torque = torch.as_tensor(
            tau_mtx[:, self._jnt_mtx2isaac], dtype=torch.float32, device=self.device
        )

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
            cur_isaac = cur[:, self._jnt_mtx2isaac]
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
            cur_isaac = cur[:, self._jnt_mtx2isaac]
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
    """Per-body ground contact via geom-pair ``is_colliding``.

    MotrixSim exposes no per-body contact force, so each (collision geom, floor) pair is
    queried and OR-aggregated per body into a boolean contact, turned into a pseudo +z
    net force, and tracked as air/contact times. Matches MjContactSensor, which also
    treats contact as a boolean (force > threshold).
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

        def zeros(*shape):
            return torch.zeros(*shape, device=self.device)

        self._data = MotrixContactData(
            net_forces_w=zeros(N, nb, 3),
            last_air_time=zeros(N, nb),
            current_air_time=zeros(N, nb),
            last_contact_time=zeros(N, nb),
            current_contact_time=zeros(N, nb),
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
    """Owns the compiled model + batched SceneData and fans calls out to the
    articulation(s) and contact sensor. Builds the combined robot + floor scene.
    """

    def __init__(self, cfg, num_envs: int, device: str, physics_dt: float, step_dt: float = 0.02):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        # step_n mode (default): advance all substeps inside one MotrixSim call with the
        # PD torque held, instead of _EnvBase's per-substep Python loop (~40x faster).
        # NO_STEPN=1 falls back to honest per-substep stepping. round() avoids the
        # int(step_dt/physics_dt) truncation bug.
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
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.name = "floor"
        g.size = [0.0, 0.0, 0.05]
        g.friction = [1.0, 0.1, 0.1]
        # Stiff floor contact (same params as the link geoms). Without it the plane uses
        # MuJoCo's soft default and the feet sink ~0.7 m through the floor. priority=1 so
        # the foot<->floor contact adopts the floor's params.
        g.solref = _CONTACT_SOLREF
        g.solimp = _CONTACT_SOLIMP
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
        except Exception as e:
            warnings.warn(f"could not set motrix solver options: {e}")
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
    """Steps the batched scene, with NaN/velocity guards and panic recovery."""

    device = "cpu"
    # Clamp only true blow-ups; high enough not to touch normal motion.
    VEL_CLAMP = 200.0

    def __init__(self, scene: MotrixScene):
        self.scene = scene
        self.model = scene.model
        self.mtx_data = scene.mtx_data

    def get_physics_dt(self):
        # In step_n mode, report step_dt so _EnvBase runs ONE step per control step;
        # the real substeps happen inside step() via step_n.
        if self.scene.use_stepn:
            return float(self.scene.step_dt)
        return float(self.model.options.timestep)

    def has_gui(self):
        return False

    def step(self, render: bool = False):
        # Scrub NaN/Inf and clamp runaway velocities before the solve so a diverging
        # env cannot blow up to NaN (which then crashes set_dof_pos at reset).
        dv = np.asarray(self.mtx_data.dof_vel)
        if (not np.isfinite(dv).all()) or (np.abs(np.nan_to_num(dv)).max() > self.VEL_CLAMP):
            dv = np.clip(np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0), -self.VEL_CLAMP, self.VEL_CLAMP)
            self.mtx_data.set_dof_vel(dv.astype(np.float32))
        try:
            if self.scene.use_stepn:
                # the engine re-evaluates the position servo each substep, so step_n runs
                # all substeps in one call with the target held.
                self.model.step_n(self.mtx_data, self.scene.substeps)
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
