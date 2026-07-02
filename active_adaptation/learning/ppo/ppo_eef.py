"""Effective impedance diagnostics for PPO policies.

This module is intentionally read-only with respect to training dynamics. It
samples rollout operating points, computes the Jacobian of the policy mean with
respect to the configured observation slices in a separate autodiff graph, and
logs effective stiffness, damping, and inertia summaries. Disabling the probe
or removing this module should not change sampled actions, log-probs, losses, or
optimizer steps.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from active_adaptation.learning.modules import VecNorm
from active_adaptation.learning.ppo.common import CMD_KEY, OBS_KEY


Tensor = torch.Tensor


@dataclass
class EffImpedanceConfig:
    """Configuration for effective impedance diagnostics.

    Args:
        enabled: Enables the diagnostic probe. The default is false.
        action_space: ``"I"`` for position target actions or ``"II"`` for
            position target plus velocity feed-forward actions.
        alpha: Action-to-position scale populated by ``ppo_symaug_eef`` from
            the current action manager's per-joint ``action_scaling``.
        beta: Velocity feed-forward coefficient for action space II.
        dt: Control period for action space II.
        q_slice: Half-open column range for joint position in the mean-network
            input tensor, populated by ``ppo_symaug_eef`` from the env
            observation group.
        qd_slice: Half-open column range for actuated joint velocity in the
            mean-network input tensor, populated by ``ppo_symaug_eef`` from the
            env observation group.
        sample_stride: Keep one operating point every ``sample_stride`` points.
        max_points: Maximum cached operating points per log window.
        log_scalars: Log scalar summaries and per-joint diagonal means.
    """

    enabled: bool = False
    action_space: str = "I"
    alpha: Any = None
    beta: float | None = None
    dt: float | None = None
    q_slice: Any = None
    qd_slice: Any = None
    sample_stride: int = 1
    max_points: int = 4096
    log_scalars: bool = True

    @classmethod
    def from_any(cls, value: Any) -> "EffImpedanceConfig":
        """Build a config from a dataclass, dict, or OmegaConf-like mapping."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        data = dict(value)
        if "alpha" in data and isinstance(data["alpha"], list):
            data["alpha"] = tuple(data["alpha"])
        for key in ("q_slice", "qd_slice"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)


def _slice(bounds: tuple[int, int], name: str) -> slice:
    start, stop = bounds
    if start < 0 or stop < start:
        raise ValueError(f"{name} must be a non-negative half-open range, got {bounds!r}")
    return slice(start, stop)


@contextmanager
def _eval_without_param_grads(module: Any):
    """Temporarily eval an nn.Module and disable parameter gradients."""

    if not isinstance(module, nn.Module):
        yield
        return

    was_training = module.training
    params = list(module.parameters())
    requires_grad = [p.requires_grad for p in params]
    module.eval()
    for param in params:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, state in zip(params, requires_grad, strict=True):
            param.requires_grad_(state)
        module.train(was_training)


def _call_mean_net(mean_net: nn.Module, obs: Tensor) -> Tensor:
    """Call a mean network on one or more observations and return loc only."""

    batched = obs.ndim > 1
    inp = obs if batched else obs.unsqueeze(0)
    out = mean_net(inp)
    if isinstance(out, TensorDictBase):
        out = out["loc"]
    elif isinstance(out, dict):
        out = out["loc"]
    elif isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError("mean_net must return a Tensor, a loc dict, or a loc TensorDict")
    return out if batched else out.squeeze(0)


def compute_policy_jacobians(
    mean_net: nn.Module,
    obs_batch: Tensor,
    cfg: EffImpedanceConfig,
) -> dict[str, Tensor]:
    """Compute policy-mean Jacobians at a batch of operating points.

    Args:
        mean_net: Module mapping observations ``(B, obs_dim)`` to action means
            ``(B, n_joints)``. It must expose the pre-sampling mean, with no
            tanh squash unless the caller has already included that derivative.
        obs_batch: Observation/input tensor with shape ``(B, obs_dim)``.
        cfg: Diagnostic config containing q and qd slices.

    Returns:
        A detached dict containing ``Jq`` with shape ``(B, n_joints, n_q)``
        and ``Jqd`` with shape ``(B, n_joints, n_qd)``.

    Notes:
        This function deliberately builds a separate autodiff graph. Parameter
        gradients are disabled and results are detached before returning.
    """

    cfg = EffImpedanceConfig.from_any(cfg)
    if obs_batch.ndim != 2:
        raise ValueError(f"obs_batch must have shape (B, obs_dim), got {tuple(obs_batch.shape)}")

    q_sl = _slice(cfg.q_slice, "q_slice")
    qd_sl = _slice(cfg.qd_slice, "qd_slice")
    obs = obs_batch.detach()

    def mean_single(single_obs: Tensor) -> Tensor:
        return _call_mean_net(mean_net, single_obs)

    with _eval_without_param_grads(mean_net):
        jac_fn = torch.func.jacrev(mean_single)
        full_jac = torch.func.vmap(jac_fn)(obs)

    return {
        "Jq": full_jac[..., q_sl].detach(),
        "Jqd": full_jac[..., qd_sl].detach(),
    }


def _gain_column(value: Any, batch: int, joints: int, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1, 1).expand(batch, joints, 1)
    if tensor.ndim == 1:
        if tensor.numel() == 1:
            return tensor.reshape(1, 1, 1).expand(batch, joints, 1)
        if tensor.numel() != joints:
            raise ValueError(f"{name} length {tensor.numel()} does not match action dim {joints}")
        return tensor.reshape(1, joints, 1).expand(batch, joints, 1)
    if tensor.ndim == 2:
        if tensor.shape == (1, joints):
            return tensor.reshape(1, joints, 1).expand(batch, joints, 1)
        if tensor.shape != (batch, joints):
            raise ValueError(f"{name} shape {tuple(tensor.shape)} must be (B, n) or (1, n)")
        return tensor.unsqueeze(-1)
    raise ValueError(f"{name} must be scalar, (n,), or (B, n), got {tuple(tensor.shape)}")


def assemble_effective_impedance(
    jacs: dict[str, Tensor],
    kp: Any,
    kd: Any,
    cfg: EffImpedanceConfig,
) -> dict[str, Tensor]:
    """Assemble effective impedance matrices from policy Jacobians.

    Args:
        jacs: Dict with ``Jq`` and ``Jqd`` tensors of shape ``(B, n, n)``.
            The qd slice must contain only actuated joints for floating-base
            robots; including base twist columns makes the damping matrix
            non-square and physically mismatched to per-joint Kd.
        kp: Scalar, ``(n,)``, or ``(B, n)`` proportional gains.
        kd: Scalar, ``(n,)``, or ``(B, n)`` derivative gains.
        cfg: Diagnostic config with action space, alpha, beta, and dt.

    Returns:
        ``Keff`` and ``Deff`` with shape ``(B, n, n)``. For action space II,
        ``Meff_delta`` is also returned.
    """

    cfg = EffImpedanceConfig.from_any(cfg)
    Jq = jacs["Jq"]
    Jqd = jacs["Jqd"]
    if Jq.ndim != 3 or Jqd.ndim != 3:
        raise ValueError("Jq and Jqd must have shape (B, n, n)")
    batch, joints, q_cols = Jq.shape
    qd_cols = Jqd.shape[-1]
    if Jqd.shape[:2] != (batch, joints):
        raise ValueError("Jq and Jqd must share batch and action dimensions")
    if q_cols != joints or qd_cols != joints:
        raise ValueError(
            f"effective impedance requires square Jq/Jqd, got Jq {tuple(Jq.shape)} "
            f"and Jqd {tuple(Jqd.shape)}"
        )

    device, dtype = Jq.device, Jq.dtype
    eye = torch.eye(joints, device=device, dtype=dtype).expand(batch, joints, joints)
    kp_col = _gain_column(kp, batch, joints, device, dtype, "kp")
    kd_col = _gain_column(kd, batch, joints, device, dtype, "kd")
    alpha_col = _gain_column(cfg.alpha, batch, joints, device, dtype, "alpha")

    alpha_Jq = alpha_col * Jq
    alpha_Jqd = alpha_col * Jqd
    Keff = kp_col * (eye - alpha_Jq)

    action_space = cfg.action_space.upper()
    if action_space == "I":
        Deff = kd_col * eye - kp_col * alpha_Jqd
        return {"Keff": Keff.detach(), "Deff": Deff.detach()}

    if action_space != "II":
        raise ValueError(f"action_space must be 'I' or 'II', got {cfg.action_space!r}")
    if cfg.beta is None or cfg.dt is None:
        raise ValueError("Action space II requires cfg.beta and cfg.dt")

    beta_dt = torch.as_tensor(float(cfg.beta) * float(cfg.dt), device=device, dtype=dtype)
    Deff = kd_col * (eye - beta_dt * alpha_Jq) - kp_col * alpha_Jqd
    Meff_delta = -kd_col * beta_dt * alpha_Jqd
    return {"Keff": Keff.detach(), "Deff": Deff.detach(), "Meff_delta": Meff_delta.detach()}


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _nominal_gain(value: Any, batch: int, joints: int) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 0:
        return np.full((batch, joints), float(arr), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 1:
            return np.full((batch, joints), float(arr.reshape(())), dtype=np.float32)
        return np.broadcast_to(arr.reshape(1, joints), (batch, joints)).astype(np.float32)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch, joints)).astype(np.float32)
        return arr.astype(np.float32)
    raise ValueError(f"gain must be scalar, (n,), or (B, n), got {arr.shape}")


def impedance_diagnostics(eff: dict[str, Tensor], kp: Any, kd: Any) -> dict[str, np.ndarray | float]:
    """Reduce effective impedance matrices to logging-friendly diagnostics.

    Args:
        eff: Dict containing ``Keff`` and ``Deff`` with shape ``(B, n, n)`` and
            optional ``Meff_delta``.
        kp: Nominal proportional gains for relative stiffness offsets.
        kd: Nominal derivative gains for relative damping offsets.

    Returns:
        A dict with per-sample/per-joint arrays and scalar stability summaries,
        including the minimum eigenvalue of the symmetric damping part and the
        fraction of sampled points where that minimum is negative.
    """

    Keff = _to_numpy(eff["Keff"])
    Deff = _to_numpy(eff["Deff"])
    batch, joints, _ = Deff.shape
    kp_nom = _nominal_gain(kp, batch, joints)
    kd_nom = _nominal_gain(kd, batch, joints)

    Kdiag = np.diagonal(Keff, axis1=-2, axis2=-1)
    Ddiag = np.diagonal(Deff, axis1=-2, axis2=-1)
    sym_K = 0.5 * (Keff + np.swapaxes(Keff, -1, -2))
    sym_D = 0.5 * (Deff + np.swapaxes(Deff, -1, -2))
    eig_K = np.linalg.eigvalsh(sym_K)
    eig_D = np.linalg.eigvalsh(sym_D)
    min_K = eig_K[:, 0]
    min_D = eig_D[:, 0]

    diagnostics: dict[str, np.ndarray | float] = {
        "Keff_diag": Kdiag,
        "Deff_diag": Ddiag,
        "Keff_diag_offset": Kdiag - kp_nom,
        "Deff_diag_offset": Ddiag - kd_nom,
        "Keff_sym_eigvals": eig_K,
        "Deff_sym_eigvals": eig_D,
        "Keff_sym_min_eig": min_K,
        "Deff_sym_min_eig": min_D,
        "Keff_sym_neg_frac": float((min_K < 0.0).mean()),
        "Deff_sym_neg_frac": float((min_D < 0.0).mean()),
        "Keff_diag_mean": float(Kdiag.mean()),
        "Deff_diag_mean": float(Ddiag.mean()),
        "Keff_diag_min": float(Kdiag.min()),
        "Deff_diag_min": float(Ddiag.min()),
        "Keff_diag_max": float(Kdiag.max()),
        "Deff_diag_max": float(Ddiag.max()),
        "Keff_sym_min_eig_mean": float(min_K.mean()),
        "Deff_sym_min_eig_mean": float(min_D.mean()),
        "Keff_sym_min_eig_min": float(min_K.min()),
        "Deff_sym_min_eig_min": float(min_D.min()),
    }
    for key in ("Meff", "Meff_delta"):
        if key in eff:
            mat = _to_numpy(eff[key])
            diag = np.diagonal(mat, axis1=-2, axis2=-1)
            diagnostics[f"{key}_diag"] = diag
            diagnostics[f"{key}_diag_mean"] = float(diag.mean())
            diagnostics[f"{key}_diag_min"] = float(diag.min())
            diagnostics[f"{key}_diag_max"] = float(diag.max())
    try:
        diagnostics["Deff_sym_cond_mean"] = float(np.linalg.cond(sym_D).mean())
        diagnostics["Keff_sym_cond_mean"] = float(np.linalg.cond(sym_K).mean())
    except np.linalg.LinAlgError:
        pass
    return diagnostics


class _PpoSymaugMean(nn.Module):
    """Mean-network adapter for ppo_symaug using raw concatenated actor input."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy
        self.vecnorm = self._find_vecnorm(policy.vecnorm)
        actor = getattr(policy, "actor")
        if isinstance(actor, nn.parallel.DistributedDataParallel):
            actor = actor.module
        self.actor_module = actor.module

    @staticmethod
    def _find_vecnorm(module: nn.Module) -> VecNorm:
        for child in module.modules():
            if isinstance(child, VecNorm):
                return child
        raise ValueError("Could not find VecNorm in policy.vecnorm")

    def forward(self, actor_input: Tensor) -> Tensor:
        normed = self.vecnorm._normalize(actor_input)
        td = TensorDict({"_obs_normed": normed}, batch_size=actor_input.shape[:-1], device=actor_input.device)
        self.actor_module(td)
        return td["loc"]


class EffImpedanceProbe:
    """Rollout-side collector and logger for effective impedance diagnostics.

    The probe caches a down-sampled set of operating points and later computes
    ``jacobians -> impedance matrices -> diagnostic scalars`` in a graph that is
    not connected to PPO optimization.
    """

    def __init__(self, cfg: EffImpedanceConfig) -> None:
        self.cfg = EffImpedanceConfig.from_any(cfg)
        self._obs_chunks: list[Tensor] = []
        self._kp_chunks: list[Tensor] = []
        self._kd_chunks: list[Tensor] = []
        self._seen = 0
        self._policy: Any | None = None

    @property
    def enabled(self) -> bool:
        """Whether this probe should collect or compute anything."""

        return bool(self.cfg.enabled)

    def sample_operating_points(
        self,
        policy: Any,
        obs: Tensor | TensorDictBase,
        kp: Any,
        kd: Any,
    ) -> None:
        """Cache down-sampled operating points from rollout or evaluation.

        Args:
            policy: PPO policy or a plain mean network. Stored only for later
                read-only Jacobian evaluation.
            obs: Tensor ``(..., obs_dim)`` or TensorDict containing ``policy``
                and optional ``command`` entries. TensorDict inputs are flattened
                into the ppo_symaug actor input order ``[command, policy]``.
            kp: Scalar, per-joint, or per-sample proportional gains.
            kd: Scalar, per-joint, or per-sample derivative gains.
        """

        if not self.enabled:
            return
        if self.num_points >= self.cfg.max_points:
            return
        obs_tensor = self._extract_actor_input(obs)
        if obs_tensor.numel() == 0:
            return

        total = obs_tensor.shape[0]
        stride = max(int(self.cfg.sample_stride), 1)
        offsets = torch.arange(total, device=obs_tensor.device) + self._seen
        keep = (offsets % stride) == 0
        self._seen += total
        idx = keep.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        remaining = self.cfg.max_points - self.num_points
        idx = idx[:remaining]
        if idx.numel() == 0:
            return

        self._policy = policy
        self._obs_chunks.append(obs_tensor.index_select(0, idx).detach().cpu())
        self._kp_chunks.append(self._select_param(kp, idx, total).detach().cpu())
        self._kd_chunks.append(self._select_param(kd, idx, total).detach().cpu())

    @property
    def num_points(self) -> int:
        """Number of cached operating points."""

        return sum(chunk.shape[0] for chunk in self._obs_chunks)

    def compute_and_log(self, logger: Any, global_step: int) -> dict[str, float]:
        """Compute diagnostics and write scalar logs.

        Args:
            logger: Either a mutable dict to update or an object with
                ``log(dict, step=...)`` such as a WandB run.
            global_step: Training/environment step used for external loggers.

        Returns:
            The scalar log dict. It is also written to ``logger``.
        """

        if not self.enabled or not self._obs_chunks:
            return {}
        if self._policy is None:
            raise RuntimeError("EffImpedanceProbe has cached observations but no policy")

        obs_cpu = torch.cat(self._obs_chunks, dim=0)
        mean_net = self._build_mean_net(self._policy)
        device = self._module_device(mean_net, obs_cpu.device)
        obs = obs_cpu.to(device=device)
        kp = self._merge_param_chunks(self._kp_chunks, obs.shape[0]).to(device=device)
        kd = self._merge_param_chunks(self._kd_chunks, obs.shape[0]).to(device=device)

        jacs = compute_policy_jacobians(mean_net, obs, self.cfg)
        eff = assemble_effective_impedance(jacs, kp, kd, self.cfg)
        diagnostics = impedance_diagnostics(eff, kp, kd)
        logs = self._scalar_logs(diagnostics, prefix="eff_impedance")

        self._write_logs(logger, logs, global_step)
        self.reset()
        return logs

    def reset(self) -> None:
        """Clear cached operating points."""

        self._obs_chunks.clear()
        self._kp_chunks.clear()
        self._kd_chunks.clear()

    @staticmethod
    def _extract_actor_input(obs: Tensor | TensorDictBase) -> Tensor:
        if torch.is_tensor(obs):
            return obs.reshape(-1, obs.shape[-1])
        keys = obs.keys(True, True)
        if CMD_KEY in keys and OBS_KEY in keys:
            tensor = torch.cat([obs[CMD_KEY], obs[OBS_KEY]], dim=-1)
        elif OBS_KEY in keys:
            tensor = obs[OBS_KEY]
        else:
            raise KeyError(f"obs TensorDict must contain {OBS_KEY!r}")
        return tensor.reshape(-1, tensor.shape[-1])

    @staticmethod
    def _select_param(value: Any, idx: Tensor, total: int) -> Tensor:
        tensor = torch.as_tensor(value, device=idx.device)
        if tensor.ndim >= 2 and tensor.shape[0] == total:
            return tensor.index_select(0, idx)
        return tensor.detach().clone()

    @staticmethod
    def _merge_param_chunks(chunks: list[Tensor], total: int) -> Tensor:
        if not chunks:
            raise ValueError("no parameter chunks to merge")
        if all(chunk.ndim >= 2 for chunk in chunks) and sum(chunk.shape[0] for chunk in chunks) == total:
            return torch.cat(chunks, dim=0)
        return chunks[-1]

    @staticmethod
    def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return fallback

    @staticmethod
    def _build_mean_net(policy: Any) -> nn.Module:
        if isinstance(policy, nn.Module) and not hasattr(policy, "vecnorm"):
            return policy
        if hasattr(policy, "vecnorm") and hasattr(policy, "actor"):
            return _PpoSymaugMean(policy)
        raise TypeError("policy must be a ppo_symaug policy or a plain nn.Module mean network")

    def _scalar_logs(self, diagnostics: dict[str, np.ndarray | float], prefix: str) -> dict[str, float]:
        if not self.cfg.log_scalars:
            return {}
        logs: dict[str, float] = {}
        for key, value in diagnostics.items():
            if isinstance(value, float):
                logs[f"{prefix}/{key}"] = value
        for name in ("Keff_diag", "Deff_diag", "Keff_diag_offset", "Deff_diag_offset"):
            value = diagnostics.get(name)
            if isinstance(value, np.ndarray) and value.ndim == 2:
                for joint_id, joint_value in enumerate(value.mean(axis=0)):
                    logs[f"{prefix}/{name}/joint_{joint_id:02d}"] = float(joint_value)
        return logs

    @staticmethod
    def _write_logs(logger: Any, logs: dict[str, float], global_step: int) -> None:
        if isinstance(logger, dict):
            logger.update(logs)
            return
        if hasattr(logger, "log"):
            try:
                logger.log(logs, step=global_step)
            except TypeError:
                logger.log(logs)
            return
        raise TypeError("logger must be a dict or expose log(dict)")
