"""Distributional RL helpers (C51-style discrete value atoms).

A small ``ValueDistribution`` bundles logits over a fixed 1-D support. That pairs
naturally with Bellman projection: the tuple is "whatever the target network
outputs at (s', a')", and :meth:`ValueDistribution.project` maps the softmax
through the categorical backup onto the same atom grid.

Keeping :func:`project_categorical_bellman` as a pure function avoids duplication
in critics (SAC, TD3, etc.) while keeping call sites explicit when you already
have loose tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import NamedTuple


def expected_q_from_logits(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Expected scalar Q under a softmax over atoms: logits [B, N] -> [B, 1]."""
    p = F.softmax(logits, dim=-1)
    z = support.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    return (p * z).sum(dim=-1, keepdim=True)


def project_categorical_bellman(
    next_logits: torch.Tensor,
    rewards: torch.Tensor,
    discount: torch.Tensor | float,
    support: torch.Tensor,
) -> torch.Tensor:
    """C51 categorical projection for a one-step (or n-step folded) Bellman backup.

    Args:
        next_logits: Target network logits at the bootstrap state, [B, num_atoms].
        rewards: Return term already folded (n-step sum, entropy in reward, etc.), [B] or [B, 1].
        discount: Factor on next-atom values; must include bootstrap mask
            (e.g. ``gamma * (1 - term)`` or :class:`MultiStepReturn` output).
        support: 1-D atom locations, shape [num_atoms], equally spaced.

    Returns:
        Projected target probabilities [B, num_atoms] (non-negative, row-stochastic).
    """
    num_atoms = support.shape[0]
    if num_atoms < 3:
        raise ValueError("support must contain more than two atoms (num_atoms > 2).")

    device = next_logits.device
    dtype = next_logits.dtype
    support = support.to(device=device, dtype=dtype)
    batch_size = next_logits.shape[0]
    if next_logits.shape[-1] != num_atoms:
        raise ValueError(
            f"next_logits last dim {next_logits.shape[-1]} != len(support) {num_atoms}"
        )

    v_lo = support[0]
    v_hi = support[-1]
    delta_raw = (v_hi - v_lo) / (num_atoms - 1)
    # Extremely small spans or dtype noise could make delta_z degenerate / unstable.
    min_delta = torch.finfo(dtype).tiny * torch.tensor(
        256.0, device=device, dtype=dtype
    )
    delta_z = torch.clamp(delta_raw, min=min_delta)

    rewards = rewards.reshape(batch_size, 1).to(dtype=dtype)
    if not isinstance(discount, torch.Tensor):
        discount_t = torch.full((batch_size, 1), float(discount), device=device, dtype=dtype)
    else:
        discount_t = discount.reshape(batch_size, 1).to(dtype=dtype)

    target_z = rewards + discount_t * support.view(1, -1)
    target_z = target_z.clamp(v_lo, v_hi)

    # Continuous index on the support grid: b=0 -> v_lo, b=num_atoms-1 -> v_hi.
    b = (target_z - v_lo) / delta_z
    b_max = float(num_atoms - 1)
    b = torch.nan_to_num(b, nan=0.0, neginf=0.0, posinf=b_max)
    b = b.clamp(0.0, b_max)

    # C51 projection: split each atom's mass between floor(b) and floor(b)+1 (Bellemare et al.).
    # Using adjacent bins avoids the old ceil/floor "same bin" hack that doubled mass on interior
    # grid points; at b = num_atoms-1, upper clamps to the same index and (1-frac)+frac preserves p.
    lower = torch.floor(b).long().clamp(0, num_atoms - 1)
    upper = (lower + 1).clamp(max=num_atoms - 1)
    frac = b - lower.to(dtype=b.dtype)

    next_dist = F.softmax(next_logits, dim=-1)
    m_l = next_dist * (1.0 - frac)
    m_u = next_dist * frac

    proj_dist = next_dist.new_zeros(batch_size, num_atoms)
    proj_dist.scatter_add_(1, lower, m_l)
    proj_dist.scatter_add_(1, upper, m_u)
    return proj_dist


class ValueDistribution(NamedTuple):
    """Softmax distribution over a fixed scalar support (e.g. C51 atoms).

    Typical use: wrap **target** logits at ``(s', a')`` and call :meth:`project`
    with bootstrapped ``rewards`` and ``discount`` to obtain a target categorical
    for cross-entropy training of the online logits at ``(s, a)``.
    """

    logits: torch.Tensor # [..., num_atoms]
    support: torch.Tensor # [num_atoms]

    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    def expected_value(self) -> torch.Tensor:
        return expected_q_from_logits(self.logits, self.support)

    def project(
        self,
        rewards: torch.Tensor,
        discount: torch.Tensor | float,
    ) -> torch.Tensor:
        """Bellman projection of ``softmax(logits)`` onto ``support``."""
        return project_categorical_bellman(
            self.logits, rewards, discount, self.support
        )
