#!/usr/bin/env python3
"""Analyze clamp saturation, energy accounting, and diagonal-damping dose response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def _load_rollout(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(npz_path) as data:
        arrays = {
            key: np.asarray(data[key]).copy()
            for key in (
                "steps",
                "tau_corr",
                "joint_vel_substep",
                "p_neg_substep",
                "s_eigvals",
                "delta_d_eigvals",
                "clamp_applied",
                "kd",
            )
        }
        meta = {
            "tau_limit": float(data["tau_limit"].item()),
            "physics_dt": float(data["physics_dt"].item()),
            "control_dt": float(data["control_dt"].item()),
            "decimation": int(data["decimation"].item()),
            "override_diag_c": float(data["override_diag_c"].item()),
            "joint_names": data["joint_names"].astype(str),
        }
    return arrays, meta


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


def _distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        str(value.item()): int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def _add_control_step_axis(axis: plt.Axes, control_dt: float) -> None:
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
    joint_names: np.ndarray,
    control_dt: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": (2, 1)})
    max_tau = np.abs(tau_sub).max(axis=-1)
    axes[0].plot(
        t_sub,
        max_tau,
        linewidth=0.8,
        color="tab:blue",
        label=r"$\max_j|\tau_{corr,j}|$",
    )
    axes[0].axhline(tau_limit, color="tab:red", linestyle="--", label="tau limit")
    axes[0].set_ylabel("torque (Nm)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_title("Clamp torque saturation at physics substeps")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    _add_control_step_axis(axes[0], control_dt)

    joint_ids = np.arange(len(joint_names))
    axes[1].bar(joint_ids, 100.0 * per_joint_rate, color="tab:blue")
    axes[1].set_xticks(joint_ids, joint_names, rotation=40, ha="right")
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
    control_dt: float,
    baseline_tau_zero: bool,
    reference_power: np.ndarray,
    reference_name: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    axes[0].plot(t_sub, np.abs(p_corr), linewidth=0.7, alpha=0.8, label=r"$|P_{corr}|$")
    axes[0].plot(t_sub, np.abs(p_neg), linewidth=0.7, alpha=0.8, label=r"$|P_{neg}|$")
    p_kd_smooth = _moving_average(np.abs(p_kd), int(round(0.1 / physics_dt)))
    axes[0].plot(t_sub, p_kd_smooth, linewidth=1.2, label=r"$|P_{Kd}|$ (0.1 s mean)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("absolute power (W)")
    axes[0].set_title("Clamp energy ledger at physics substeps")
    axes[0].grid(alpha=0.25)
    lam_axis = axes[0].twinx()
    lam_axis.plot(
        lam_time,
        lam_min,
        color="black",
        linewidth=0.65,
        alpha=0.55,
        label=r"$\min\,\mathrm{eig}(\mathrm{sym}(D_{eff}))$",
    )
    lam_axis.set_ylabel(r"$\min\,\mathrm{eig}(\mathrm{sym}(D_{eff}))$")
    lines = axes[0].get_lines() + lam_axis.get_lines()
    axes[0].legend(lines, [line.get_label() for line in lines], loc="upper right")
    _add_control_step_axis(axes[0], control_dt)

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
        x = reference_power[trim_mask]
        y = p_corr[trim_mask]
        sat = saturated_substep[trim_mask]
        axes[1].scatter(x[~sat], y[~sat], s=8, alpha=0.35, label="not saturated")
        axes[1].scatter(
            x[sat],
            y[sat],
            s=10,
            alpha=0.55,
            color="tab:red",
            label="saturated",
        )
        line_min = min(float(x.min()), float(y.min()))
        line_max = max(float(x.max()), float(y.max()))
        axes[1].plot(
            [line_min, line_max],
            [line_min, line_max],
            "k--",
            linewidth=1.0,
            label="y = x",
        )
        axes[1].legend(loc="best")
    axes[1].set_xlabel(f"{reference_name} (W)")
    axes[1].set_ylabel(r"$P_{corr}$ (W)")
    axes[1].set_title(f"Applied versus {reference_name} power after trim")
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
        f"mode: {summary['mode']}",
        f"override_diag_c: {_format_value(summary['override_diag_c'])}",
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
        ]
    )
    override_accounting = summary["override_accounting"]
    if override_accounting is not None:
        lines.extend(
            [
                "",
                "override_accounting_unsaturated:",
                f"  mean_abs_p_expected: {_format_value(override_accounting['mean_abs_p_expected'])}",
                f"  ratio_corr_expected: {_format_value(override_accounting['ratio_corr_expected'])}",
                f"  residual_mean: {_format_value(override_accounting['residual_mean'])}",
                f"  residual_std: {_format_value(override_accounting['residual_std'])}",
                f"  max_relative_error: {_format_value(override_accounting['max_relative_error'])}",
            ]
        )
    lines.extend(
        [
            "",
            "lam_min:",
            f"  p5: {_format_value(lam_min['p5'])}",
            f"  p50: {_format_value(lam_min['p50'])}",
            f"  p95: {_format_value(lam_min['p95'])}",
        ]
    )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, object]:
    arrays, meta = _load_rollout(args.npz_path)
    if args.trim_start_s < 0.0:
        raise ValueError(f"trim_start_s must be non-negative, got {args.trim_start_s}")

    tau_limit = meta["tau_limit"]
    physics_dt = meta["physics_dt"]
    control_dt = meta["control_dt"]
    decimation = meta["decimation"]
    override_diag_c = meta["override_diag_c"]
    joint_names = meta["joint_names"]

    distribution = _distribution(arrays["clamp_applied"])
    print(f"clamp_applied distribution: {distribution}")
    if np.all(arrays["clamp_applied"] == 0):
        print("clamp_applied classification: baseline rollout (all 0)")
        mode = "baseline"
    elif np.all(arrays["clamp_applied"] == 1):
        print("clamp_applied classification: clamp rollout (all 1)")
        mode = "override_diag" if override_diag_c > 0.0 else "clamp"
    else:
        print("warning: clamp_applied contains mixed values")
        mode = "mixed"

    num_steps = arrays["steps"].shape[0]
    joints = arrays["tau_corr"].shape[-1]
    tau_sub = arrays["tau_corr"].reshape(num_steps * decimation, joints)
    joint_vel_sub = arrays["joint_vel_substep"].reshape(num_steps * decimation, joints)
    p_neg = arrays["p_neg_substep"].reshape(num_steps * decimation)
    kd_sub = np.repeat(arrays["kd"], decimation, axis=0)
    t_sub = np.arange(num_steps * decimation, dtype=np.float64) * physics_dt
    lam_time = np.arange(num_steps, dtype=np.float64) * control_dt
    trim_mask = t_sub >= args.trim_start_s
    trim_control_mask = lam_time >= args.trim_start_s
    if not trim_mask.any() or not trim_control_mask.any():
        raise ValueError(
            f"trim_start_s={args.trim_start_s} removes the entire "
            f"{num_steps * control_dt:.6g}s rollout"
        )

    sat_mask = np.abs(tau_sub) >= 0.98 * tau_limit
    saturated_substep = sat_mask.any(axis=-1)
    p_corr = np.sum(tau_sub * joint_vel_sub, axis=-1)
    p_kd = -np.sum(kd_sub * joint_vel_sub**2, axis=-1)
    p_override = -override_diag_c * np.sum(joint_vel_sub**2, axis=-1)
    lam_min = arrays["s_eigvals"].min(axis=-1)

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
    override_accounting = None
    if override_diag_c > 0.0:
        unsaturated = ~saturated_trimmed
        applied = p_corr_trimmed[unsaturated]
        expected = p_override[trim_mask][unsaturated]
        relative_error = np.abs(applied - expected) / np.maximum(np.abs(applied), 1.0e-6)
        override_accounting = {
            "mean_abs_p_expected": float(np.abs(expected).mean()),
            "ratio_corr_expected": _ratio(
                float(np.abs(applied).mean()),
                float(np.abs(expected).mean()),
            ),
            "residual_mean": float((applied - expected).mean()),
            "residual_std": float((applied - expected).std()),
            "max_relative_error": float(relative_error.max()),
        }

    summary: dict[str, object] = {
        "roll_name": args.npz_path.parent.parent.name,
        "npz_path": str(args.npz_path.resolve()),
        "mode": mode,
        "override_diag_c": float(override_diag_c),
        "T": int(num_steps),
        "duration_s": float(num_steps * control_dt),
        "tau_limit": float(tau_limit),
        "physics_dt": float(physics_dt),
        "kd": float(arrays["kd"].mean()),
        "trim_start_s": float(args.trim_start_s),
        "clamp_applied_distribution": distribution,
        "saturation": {
            "per_joint_rate": {
                name: float(rate)
                for name, rate in zip(joint_names, per_joint_rate, strict=True)
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
        "override_accounting": override_accounting,
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
        joint_names,
        control_dt,
    )
    reference_power = p_override if override_diag_c > 0.0 else p_neg
    reference_name = r"$P_{diag}$" if override_diag_c > 0.0 else r"$P_{neg}$"
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
        control_dt,
        baseline_tau_zero=bool(np.all(tau_sub == 0.0)),
        reference_power=reference_power,
        reference_name=reference_name,
    )
    _write_summary(out_dir, summary)

    saturation_rate = summary["saturation"]["global_rate"]
    ratio_neg_pd = summary["energy"]["ratio_neg_pd"]
    print(
        f"[hint] saturation rate={100.0 * saturation_rate:.4g}% "
        "(reference threshold: 5%)"
    )
    if override_accounting is not None:
        print(
            "[hint] ratio_corr_override="
            f"{_format_value(override_accounting['ratio_corr_expected'])} "
            "(reference range: 0.9-1.1)"
        )
    else:
        ratio_corr_neg = summary["energy"]["ratio_corr_neg"]
        print(
            f"[hint] ratio_corr_neg={_format_value(ratio_corr_neg)} "
            "(reference range: 0.9-1.1)"
        )
    print(
        f"[hint] ratio_neg_pd={_format_value(ratio_neg_pd)} "
        "(reference threshold: 0.05)"
    )
    return summary


def _dose_record(
    npz_path: Path,
    trim_start_s: float,
) -> dict[str, object]:
    arrays, meta = _load_rollout(npz_path)
    tau_limit = meta["tau_limit"]
    physics_dt = meta["physics_dt"]
    control_dt = meta["control_dt"]
    decimation = meta["decimation"]
    override_diag_c = meta["override_diag_c"]
    joint_names = meta["joint_names"]
    num_steps = arrays["steps"].shape[0]
    joints = arrays["joint_vel_substep"].shape[-1]
    joint_vel = arrays["joint_vel_substep"].reshape(num_steps * decimation, joints)
    tau_corr = arrays["tau_corr"].reshape(num_steps * decimation, joints)
    time = np.arange(num_steps * decimation, dtype=np.float64) * physics_dt
    trim_mask = time >= trim_start_s
    if not trim_mask.any():
        raise ValueError(
            f"trim_start_s={trim_start_s} removes the entire "
            f"{num_steps * control_dt:.6g}s rollout {npz_path}"
        )

    joint_vel = joint_vel[trim_mask]
    tau_corr = tau_corr[trim_mask]
    per_joint_rms = np.sqrt(np.mean(joint_vel**2, axis=0))
    combined_rms = float(np.sqrt(np.mean(joint_vel**2)))
    frequencies, psd = welch(
        joint_vel,
        fs=1.0 / physics_dt,
        window="hann",
        nperseg=min(2048, joint_vel.shape[0]),
        detrend="constant",
        axis=0,
    )
    mean_psd = psd.mean(axis=1)
    peak_band = (frequencies >= 0.5) & (frequencies <= 25.0)
    peak_frequency = float(frequencies[peak_band][np.argmax(mean_psd[peak_band])])
    p_corr = np.sum(tau_corr * joint_vel, axis=-1)
    saturation_rate = float((np.abs(tau_corr) >= 0.98 * tau_limit).mean())

    return {
        "roll_name": npz_path.parent.parent.name,
        "npz_path": str(npz_path.resolve()),
        "override_diag_c": float(override_diag_c),
        "tau_limit": float(tau_limit),
        "physics_dt": float(physics_dt),
        "T": int(num_steps),
        "duration_s": float(num_steps * control_dt),
        "trim_start_s": float(trim_start_s),
        "joint_vel_rms": combined_rms,
        "joint_vel_rms_per_joint": {
            name: float(value)
            for name, value in zip(joint_names, per_joint_rms, strict=True)
        },
        "joint_vel_peak_hz": peak_frequency,
        "mean_abs_p_corr": float(np.abs(p_corr).mean()),
        "saturation_rate": saturation_rate,
    }


def _plot_dose_response(
    out_path: Path,
    records: list[dict[str, object]],
) -> None:
    doses = np.asarray([record["override_diag_c"] for record in records], dtype=float)
    rms = np.asarray([record["joint_vel_rms"] for record in records], dtype=float)
    peak_hz = np.asarray([record["joint_vel_peak_hz"] for record in records], dtype=float)
    saturation = np.asarray([record["saturation_rate"] for record in records], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(doses, rms, marker="o", linewidth=1.5, color="tab:blue")
    for dose, value, record in zip(doses, rms, records, strict=True):
        axes[0].annotate(
            str(record["roll_name"]),
            (dose, value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_ylabel(r"joint velocity RMS (rad/s)")
    axes[0].set_title("Suspended diagonal-damping dose response")
    axes[0].grid(alpha=0.25)
    saturation_axis = axes[0].twinx()
    bar_width = 0.02 * max(float(np.ptp(doses)), 1.0)
    saturation_axis.bar(
        doses,
        100.0 * saturation,
        width=bar_width,
        alpha=0.25,
        color="tab:red",
        label="saturation rate",
    )
    saturation_axis.set_ylabel("saturation rate (%)")

    axes[1].plot(doses, peak_hz, marker="o", linewidth=1.5, color="tab:orange")
    for dose, value, record in zip(doses, peak_hz, records, strict=True):
        axes[1].annotate(
            str(record["roll_name"]),
            (dose, value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel(r"override diagonal damping $c$ (Nms/rad)")
    axes[1].set_ylabel("dominant joint-velocity frequency (Hz)")
    axes[1].set_title("Mean Welch-PSD peak in 0.5-25 Hz")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_dose(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.trim_start_s < 0.0:
        raise ValueError(f"trim_start_s must be non-negative, got {args.trim_start_s}")
    records = [
        _dose_record(path, trim_start_s=args.trim_start_s)
        for path in args.dose
    ]
    records.sort(key=lambda record: record["override_diag_c"])

    out_dir = args.out_dir or args.dose[0].parent / "dose"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_dose_response(out_dir / "dose_response.png", records)
    with (out_dir / "dose_summary.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, allow_nan=False)
        file.write("\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz_path",
        nargs="?",
        type=Path,
        help="Path to one eff_impedance_timeseries.npz",
    )
    parser.add_argument(
        "--dose",
        nargs="+",
        type=Path,
        help="Aggregate diagonal-damping dose-response NPZ files",
    )
    parser.add_argument(
        "--trim-start-s",
        type=float,
        default=2.0,
        help="Seconds excluded from scalar summaries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory (default: <NPZ directory>/analysis or dose)",
    )
    args = parser.parse_args()
    if args.dose is not None and args.npz_path is not None:
        parser.error("provide either one positional NPZ or --dose NPZ files, not both")
    if args.dose is None and args.npz_path is None:
        parser.error("provide one positional NPZ or --dose NPZ files")
    try:
        if args.dose is not None:
            analyze_dose(args)
        else:
            analyze(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
