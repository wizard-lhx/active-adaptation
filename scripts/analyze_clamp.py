#!/usr/bin/env python3
"""Analyze clamp saturation and energy accounting from a recorded NPZ rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_KEYS = (
    "tau_corr",
    "joint_vel_substep",
    "p_neg_substep",
    "delta_d_eigvals",
    "clamp_applied",
    "steps",
)

# B2 action order from active_adaptation/assets/quadrupeds/b2_manipulatior.py:37.
JOINT_NAMES = (
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)


def _finite_npz_scalar(data: np.lib.npyio.NpzFile, key: str, fallback: float) -> float:
    if key not in data.files:
        return float(fallback)
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"NPZ key {key!r} must be a scalar, got shape {value.shape}")
    scalar = float(value.reshape(-1)[0])
    return scalar if np.isfinite(scalar) else float(fallback)


def _load_rollout(
    npz_path: Path,
    tau_limit_fallback: float,
    physics_dt_fallback: float,
) -> tuple[dict[str, np.ndarray], float, float]:
    with np.load(npz_path) as data:
        available = list(data.files)
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"missing required NPZ keys: {missing}; actual keys: {available}"
            )
        arrays = {key: np.asarray(data[key]).copy() for key in REQUIRED_KEYS}
        tau_limit = _finite_npz_scalar(data, "tau_limit", tau_limit_fallback)
        physics_dt = _finite_npz_scalar(data, "physics_dt", physics_dt_fallback)

    steps = arrays["steps"]
    if steps.ndim != 1 or steps.size == 0:
        raise ValueError(f"steps must have shape (T,) with T > 0, got {steps.shape}")
    T = steps.shape[0]
    expected_shapes = {
        "tau_corr": (T, 4, 12),
        "joint_vel_substep": (T, 4, 12),
        "p_neg_substep": (T, 4),
        "delta_d_eigvals": (T, 12),
        "clamp_applied": (T,),
        "steps": (T,),
    }
    bad_shapes = [
        f"{key}: expected {shape}, got {arrays[key].shape}"
        for key, shape in expected_shapes.items()
        if arrays[key].shape != shape
    ]
    if bad_shapes:
        raise ValueError("invalid NPZ shapes: " + "; ".join(bad_shapes))
    if T > 1 and not np.all(np.diff(steps) == 1):
        raise ValueError(
            "steps must be a consecutive sequence with increment 1; "
            "update_interval>1 caused holes in clamp records, so this rollout cannot be used "
            f"for energy accounting (first={steps[0]}, last={steps[-1]})"
        )
    if tau_limit <= 0.0:
        raise ValueError(f"tau_limit must be positive, got {tau_limit}")
    if physics_dt <= 0.0:
        raise ValueError(f"physics_dt must be positive, got {physics_dt}")
    return arrays, tau_limit, physics_dt


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = min(max(window, 1), values.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _group_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator != 0.0 else None


def _add_control_step_axis(axis: plt.Axes, physics_dt: float) -> None:
    control_dt = 4.0 * physics_dt
    secondary = axis.secondary_xaxis(
        "top",
        functions=(lambda seconds: seconds / control_dt, lambda step: step * control_dt),
    )
    secondary.set_xlabel("control step")


def _plot_saturation(
    out_path: Path,
    t_sub: np.ndarray,
    tau_sub: np.ndarray,
    tau_limit: float,
    per_joint_rate: np.ndarray,
    physics_dt: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": (2, 1)})
    max_tau = np.abs(tau_sub).max(axis=-1)
    axes[0].plot(t_sub, max_tau, linewidth=0.8, color="tab:blue", label=r"$\max_j|\tau_{corr,j}|$")
    axes[0].axhline(tau_limit, color="tab:red", linestyle="--", label="tau limit")
    axes[0].set_ylabel("torque (Nm)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_title("Clamp torque saturation at 200 Hz")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    _add_control_step_axis(axes[0], physics_dt)

    joint_ids = np.arange(len(JOINT_NAMES))
    axes[1].bar(joint_ids, 100.0 * per_joint_rate, color="tab:blue")
    axes[1].set_xticks(joint_ids, JOINT_NAMES, rotation=40, ha="right")
    axes[1].set_ylabel("saturated substeps (%)")
    axes[1].set_title("Per-joint saturation rate after trim")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_energy_ledger(
    out_path: Path,
    t_sub: np.ndarray,
    p_corr: np.ndarray,
    p_neg: np.ndarray,
    p_kd: np.ndarray,
    lam_time: np.ndarray,
    lam_min: np.ndarray,
    trim_mask: np.ndarray,
    saturated_substep: np.ndarray,
    physics_dt: float,
    baseline_tau_zero: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    axes[0].plot(t_sub, np.abs(p_corr), linewidth=0.7, alpha=0.8, label=r"$|P_{corr}|$")
    axes[0].plot(t_sub, np.abs(p_neg), linewidth=0.7, alpha=0.8, label=r"$|P_{neg}|$")
    p_kd_smooth = _moving_average(np.abs(p_kd), int(round(0.1 / physics_dt)))
    axes[0].plot(t_sub, p_kd_smooth, linewidth=1.2, label=r"$|P_{Kd}|$ (0.1 s mean)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("absolute power (W)")
    axes[0].set_title("Clamp energy ledger at 200 Hz")
    axes[0].grid(alpha=0.25)
    lam_axis = axes[0].twinx()
    lam_axis.plot(
        lam_time,
        lam_min,
        color="black",
        linewidth=0.65,
        alpha=0.55,
        label=r"$\min\,\mathrm{eig}(\Delta D)$",
    )
    lam_axis.set_ylabel(r"$\min\,\mathrm{eig}(\Delta D)$")
    lines = axes[0].get_lines() + lam_axis.get_lines()
    axes[0].legend(lines, [line.get_label() for line in lines], loc="upper right")
    _add_control_step_axis(axes[0], physics_dt)

    if baseline_tau_zero:
        axes[1].text(
            0.5,
            0.5,
            "Baseline rollout: tau_corr is zero; scatter skipped.",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        x = p_neg[trim_mask]
        y = p_corr[trim_mask]
        sat = saturated_substep[trim_mask]
        axes[1].scatter(x[~sat], y[~sat], s=8, alpha=0.35, label="not saturated")
        axes[1].scatter(x[sat], y[sat], s=10, alpha=0.55, color="tab:red", label="saturated")
        line_min = min(float(x.min()), float(y.min()))
        line_max = max(float(x.max()), float(y.max()))
        axes[1].plot([line_min, line_max], [line_min, line_max], "k--", linewidth=1.0, label="y = x")
        axes[1].legend(loc="best")
    axes[1].set_xlabel(r"$P_{neg}$ (W)")
    axes[1].set_ylabel(r"$P_{corr}$ (W)")
    axes[1].set_title("Applied versus theoretical negative-mode power after trim")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _format_value(value: float | int | None, precision: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}g}"


def _write_summary(out_dir: Path, summary: dict[str, object]) -> None:
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")

    saturation = summary["saturation"]
    energy = summary["energy"]
    residual = summary["residual"]
    lam_min = summary["lam_min"]
    lines = [
        f"roll_name: {summary['roll_name']}",
        f"npz_path: {summary['npz_path']}",
        f"T: {summary['T']}",
        f"duration_s: {_format_value(summary['duration_s'])}",
        f"tau_limit: {_format_value(summary['tau_limit'])}",
        f"physics_dt: {_format_value(summary['physics_dt'])}",
        f"kd: {_format_value(summary['kd'])}",
        f"trim_start_s: {_format_value(summary['trim_start_s'])}",
        f"clamp_applied_distribution: {summary['clamp_applied_distribution']}",
        "",
        "saturation:",
        f"  global_rate: {_format_value(saturation['global_rate'])}",
        f"  abs_tau_p50: {_format_value(saturation['abs_tau_p50'])}",
        f"  abs_tau_p90: {_format_value(saturation['abs_tau_p90'])}",
        f"  abs_tau_p99: {_format_value(saturation['abs_tau_p99'])}",
        f"  abs_tau_max: {_format_value(saturation['abs_tau_max'])}",
        "  per_joint_rate:",
    ]
    lines.extend(
        f"    {name}: {_format_value(rate)}"
        for name, rate in saturation["per_joint_rate"].items()
    )
    lines.extend(
        [
            "",
            "energy:",
            f"  mean_abs_p_corr: {_format_value(energy['mean_abs_p_corr'])}",
            f"  mean_abs_p_neg: {_format_value(energy['mean_abs_p_neg'])}",
            f"  mean_abs_p_kd: {_format_value(energy['mean_abs_p_kd'])}",
            f"  ratio_corr_neg: {_format_value(energy['ratio_corr_neg'])}",
            f"  ratio_neg_pd: {_format_value(energy['ratio_neg_pd'])}",
            f"  frac_pcorr_pos: {_format_value(energy['frac_pcorr_pos'])}",
            "",
            "residual_p_corr_minus_p_neg:",
            "  all:",
            f"    count: {_format_value(residual['all']['count'])}",
            f"    mean: {_format_value(residual['all']['mean'])}",
            f"    std: {_format_value(residual['all']['std'])}",
            "  saturated:",
            f"    count: {_format_value(residual['saturated']['count'])}",
            f"    mean: {_format_value(residual['saturated']['mean'])}",
            f"    std: {_format_value(residual['saturated']['std'])}",
            "  unsaturated:",
            f"    count: {_format_value(residual['unsaturated']['count'])}",
            f"    mean: {_format_value(residual['unsaturated']['mean'])}",
            f"    std: {_format_value(residual['unsaturated']['std'])}",
            "",
            "lam_min:",
            f"  p5: {_format_value(lam_min['p5'])}",
            f"  p50: {_format_value(lam_min['p50'])}",
            f"  p95: {_format_value(lam_min['p95'])}",
        ]
    )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, object]:
    arrays, tau_limit, physics_dt = _load_rollout(
        args.npz_path,
        tau_limit_fallback=args.tau_limit,
        physics_dt_fallback=args.physics_dt,
    )
    if args.kd < 0.0:
        raise ValueError(f"kd must be non-negative, got {args.kd}")
    if args.trim_start_s < 0.0:
        raise ValueError(f"trim_start_s must be non-negative, got {args.trim_start_s}")

    clamp_values, clamp_counts = np.unique(arrays["clamp_applied"], return_counts=True)
    distribution = {
        str(value.item()): int(count)
        for value, count in zip(clamp_values, clamp_counts, strict=True)
    }
    print(f"clamp_applied distribution: {distribution}")
    if np.all(arrays["clamp_applied"] == 0):
        print("clamp_applied classification: baseline rollout (all 0)")
    elif np.all(arrays["clamp_applied"] == 1):
        print("clamp_applied classification: clamp rollout (all 1)")
    else:
        print("warning: clamp_applied contains mixed values")

    T = arrays["steps"].shape[0]
    tau_sub = arrays["tau_corr"].reshape(T * 4, 12)
    joint_vel_sub = arrays["joint_vel_substep"].reshape(T * 4, 12)
    p_neg = arrays["p_neg_substep"].reshape(T * 4)
    t_sub = np.arange(T * 4, dtype=np.float64) * physics_dt
    lam_time = np.arange(T, dtype=np.float64) * 4.0 * physics_dt
    trim_mask = t_sub >= args.trim_start_s
    trim_control_mask = lam_time >= args.trim_start_s
    if not trim_mask.any() or not trim_control_mask.any():
        raise ValueError(
            f"trim_start_s={args.trim_start_s} removes the entire {T * 4 * physics_dt:.6g}s rollout"
        )

    sat_mask = np.abs(tau_sub) >= 0.98 * tau_limit
    saturated_substep = sat_mask.any(axis=-1)
    p_corr = np.sum(tau_sub * joint_vel_sub, axis=-1)
    p_kd = -args.kd * np.sum(joint_vel_sub**2, axis=-1)
    lam_min = arrays["delta_d_eigvals"].min(axis=-1)

    tau_trimmed = tau_sub[trim_mask]
    sat_trimmed = sat_mask[trim_mask]
    p_corr_trimmed = p_corr[trim_mask]
    p_neg_trimmed = p_neg[trim_mask]
    p_kd_trimmed = p_kd[trim_mask]
    saturated_trimmed = saturated_substep[trim_mask]
    residual_trimmed = p_corr_trimmed - p_neg_trimmed
    lam_trimmed = lam_min[trim_control_mask]

    per_joint_rate = sat_trimmed.mean(axis=0)
    mean_abs_p_corr = float(np.abs(p_corr_trimmed).mean())
    mean_abs_p_neg = float(np.abs(p_neg_trimmed).mean())
    mean_abs_p_kd = float(np.abs(p_kd_trimmed).mean())
    abs_tau = np.abs(tau_trimmed).reshape(-1)
    summary: dict[str, object] = {
        "roll_name": args.npz_path.parent.parent.name,
        "npz_path": str(args.npz_path.resolve()),
        "T": int(T),
        "duration_s": float(T * 4 * physics_dt),
        "tau_limit": float(tau_limit),
        "physics_dt": float(physics_dt),
        "kd": float(args.kd),
        "trim_start_s": float(args.trim_start_s),
        "clamp_applied_distribution": distribution,
        "saturation": {
            "per_joint_rate": {
                name: float(rate) for name, rate in zip(JOINT_NAMES, per_joint_rate, strict=True)
            },
            "global_rate": float(sat_trimmed.mean()),
            "abs_tau_p50": float(np.percentile(abs_tau, 50)),
            "abs_tau_p90": float(np.percentile(abs_tau, 90)),
            "abs_tau_p99": float(np.percentile(abs_tau, 99)),
            "abs_tau_max": float(abs_tau.max()),
        },
        "energy": {
            "mean_abs_p_corr": mean_abs_p_corr,
            "mean_abs_p_neg": mean_abs_p_neg,
            "mean_abs_p_kd": mean_abs_p_kd,
            "ratio_corr_neg": _ratio(mean_abs_p_corr, mean_abs_p_neg),
            "ratio_neg_pd": _ratio(mean_abs_p_neg, mean_abs_p_kd),
            "frac_pcorr_pos": float((p_corr_trimmed > 1.0e-8).mean()),
        },
        "residual": {
            "all": _group_stats(residual_trimmed),
            "saturated": _group_stats(residual_trimmed[saturated_trimmed]),
            "unsaturated": _group_stats(residual_trimmed[~saturated_trimmed]),
        },
        "lam_min": {
            "p5": float(np.percentile(lam_trimmed, 5)),
            "p50": float(np.percentile(lam_trimmed, 50)),
            "p95": float(np.percentile(lam_trimmed, 95)),
        },
    }

    out_dir = args.out_dir or args.npz_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_saturation(
        out_dir / "saturation.png",
        t_sub,
        tau_sub,
        tau_limit,
        per_joint_rate,
        physics_dt,
    )
    _plot_energy_ledger(
        out_dir / "energy_ledger.png",
        t_sub,
        p_corr,
        p_neg,
        p_kd,
        lam_time,
        lam_min,
        trim_mask,
        saturated_substep,
        physics_dt,
        baseline_tau_zero=bool(np.all(tau_sub == 0.0)),
    )
    _write_summary(out_dir, summary)

    saturation_rate = summary["saturation"]["global_rate"]
    ratio_corr_neg = summary["energy"]["ratio_corr_neg"]
    ratio_neg_pd = summary["energy"]["ratio_neg_pd"]
    print(f"[hint] saturation rate={100.0 * saturation_rate:.4g}% (reference threshold: 5%)")
    print(f"[hint] ratio_corr_neg={_format_value(ratio_corr_neg)} (reference range: 0.9-1.1)")
    print(f"[hint] ratio_neg_pd={_format_value(ratio_neg_pd)} (reference threshold: 0.05)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Path to eff_impedance_timeseries.npz")
    parser.add_argument("--tau-limit", type=float, required=True, help="Fallback clamp torque limit in Nm")
    parser.add_argument("--physics-dt", type=float, default=0.005, help="Fallback physics timestep in seconds")
    parser.add_argument("--kd", type=float, default=2.0, help="Nominal joint damping used for P_kd")
    parser.add_argument("--trim-start-s", type=float, default=2.0, help="Seconds excluded from scalar summaries")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: <npz directory>/analysis)")
    args = parser.parse_args()
    try:
        analyze(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
