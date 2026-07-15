"""Offline negative-damping mode identification.

Mode consistency identifies whether the minimum-eigenvalue direction persists,
while peak frequency and stride-band energy identify where that mode injects
energy. The two-stage verdict uses consistency first, then spectral energy:
HAZARD suggests raising Kd from 2 to 7--8 or reducing action scale; ENGINE can
be checked against a milder Kd increase from 2 to 4--5.

Known limitations: the effective-impedance formula is a static approximation
that omits the action LPF (alpha_range=[0.8, 0.9]); Kp/Kd are randomized in
[0.8, 1.1], and num_envs=1 uses that environment's realized gains.
"""

import numpy as np


def compute_negative_mode(
    Deff,
    joint_vel,
    done,
    dt,
    t_skip=50,
    degeneracy_rel=0.1,
    stride_band=(0.5, 5.0),
) -> dict:
    """Compute the aligned minimum mode and its temporal/spectral diagnostics."""

    sym_deff = 0.5 * (Deff + np.swapaxes(Deff, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(sym_deff)
    lam_min = eigvals[:, 0]
    v_min = eigvecs[:, :, 0].copy()

    for t in range(1, len(v_min)):
        if not done[t - 1] and np.dot(v_min[t], v_min[t - 1]) < 0.0:
            v_min[t] *= -1.0

    degenerate = (eigvals[:, 1] - eigvals[:, 0]) < (
        degeneracy_rel * np.maximum(np.abs(eigvals[:, 0]), 1e-9)
    )
    consistency = np.abs(np.sum(v_min[:-1] * v_min[1:], axis=1))
    consistency[done[:-1] | degenerate[:-1] | degenerate[1:]] = np.nan

    valid = ~degenerate
    segment_starts = np.r_[0, np.flatnonzero(done[:-1]) + 1]
    segment_ends = np.r_[segment_starts[1:], len(done)]
    for start, end in zip(segment_starts, segment_ends, strict=True):
        valid[start : min(start + t_skip, end)] = False

    consistency_mask = valid[:-1] & valid[1:] & ~done[:-1]
    median_consistency = float(np.median(consistency[consistency_mask]))
    mode_velocity = np.sum(v_min * joint_vel, axis=1)

    edges = np.diff(np.pad(valid.astype(np.int8), (1, 1)))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    longest = np.argmax(run_ends - run_starts)
    signal = mode_velocity[run_starts[longest] : run_ends[longest]]
    signal = signal - signal.mean()
    windowed = signal * np.hanning(len(signal))
    spectrum = np.fft.rfft(windowed)
    psd = np.abs(spectrum) ** 2 / np.sum(np.hanning(len(signal)) ** 2)
    psd_f = np.fft.rfftfreq(len(signal), d=dt)
    f_peak = float(psd_f[1 + np.argmax(psd[1:])])
    in_band = (psd_f >= stride_band[0]) & (psd_f <= stride_band[1])
    band_frac = float(psd[in_band].sum() / psd[1:].sum())

    return {
        "v_min": v_min,
        "lam_min": lam_min,
        "consistency": consistency,
        "median_consistency": median_consistency,
        "f_peak": f_peak,
        "band_frac": band_frac,
        "psd_f": psd_f,
        "psd": psd,
        "valid": valid,
    }


def verdict(
    median_consistency,
    f_peak,
    band_frac,
    psd_f,
    psd,
    stride_band=(0.5, 5.0),
) -> str:
    """Classify a negative mode using persistence, then spectral energy."""

    if median_consistency < 0.6:
        return "HAZARD"
    if median_consistency <= 0.9:
        return "MIXED"
    if stride_band[0] <= f_peak <= stride_band[1] and band_frac > 0.5:
        return "ENGINE"
    in_band = (psd_f >= stride_band[0]) & (psd_f <= stride_band[1])
    high_frequency = psd_f > 2.0 * stride_band[1]
    if psd[high_frequency].sum() > psd[in_band].sum():
        return "HAZARD"
    return "MIXED"


def contact_count_lambda_medians(lam_min, contact, valid) -> np.ndarray:
    """Return valid-frame lambda-min medians for 0 through 4 contacting feet."""

    contact_count = np.asarray(contact).sum(axis=1)
    buckets = [(contact_count == count) & valid for count in range(5)]
    return np.asarray(
        [np.median(lam_min[bucket]) if bucket.any() else np.nan for bucket in buckets]
    )
