"""Replay video with augmented stiffness differences and foot contact phases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from visualize_eff_impedance import (
    LEG_JOINT_NAMES,
    LEG_NAMES,
    Viewer,
    configure_joint_matrix_axis,
    leg_joint_permutation,
    reorder_joint_matrix,
)


class DeltaKViewer(Viewer):
    """Visualize absolute stiffness deltas and global/per-leg relative norms."""

    _colors = {
        "Global": "black",
        "FL": "tab:blue",
        "FR": "tab:orange",
        "RL": "tab:green",
        "RR": "tab:red",
    }

    def _load_data(self, npz_path: Path) -> None:
        with np.load(npz_path) as data:
            required = {
                "steps",
                "Keff",
                "Keff_aug",
                "Jdot_valid",
                "joint_names",
                "foot_contact",
                "foot_names",
                "control_dt",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise KeyError(f"Recording is missing augmented visualization keys: {missing}")

            self.steps = data["steps"].copy()
            joint_names = data["joint_names"].astype(str)
            permutation = leg_joint_permutation(joint_names)
            self.joint_names = joint_names[permutation]
            self.Keff = reorder_joint_matrix(data["Keff"], permutation)
            self.Keff_aug = reorder_joint_matrix(data["Keff_aug"], permutation)
            self.jdot_valid = data["Jdot_valid"].astype(bool)
            self.foot_contact = data["foot_contact"].astype(bool)
            self.foot_names = data["foot_names"].astype(str)
            self.control_dt = float(data["control_dt"])

        if tuple(self.joint_names) != LEG_JOINT_NAMES:
            raise ValueError(f"Expected B2 leg joints in leg-major order, got {self.joint_names}.")
        if self.foot_contact.shape != (len(self.steps), len(LEG_NAMES)):
            raise ValueError(
                f"Expected foot_contact shape ({len(self.steps)}, {len(LEG_NAMES)}), "
                f"got {self.foot_contact.shape}."
            )

        self.times = self.steps * self.control_dt
        self.delta_k = np.abs(self.Keff_aug - self.Keff)
        finite = (
            np.isfinite(self.Keff).all(axis=(-2, -1))
            & np.isfinite(self.delta_k).all(axis=(-2, -1))
        )
        self.valid = self.jdot_valid & finite
        if not self.valid.any():
            raise ValueError("Recording has no finite frames with Jdot_valid=True.")

        self.ratio_rows = {
            "Global": slice(None),
            "FL": slice(0, 3),
            "FR": slice(3, 6),
            "RL": slice(6, 9),
            "RR": slice(9, 12),
        }
        self.ratios = {
            name: self._relative_norm(rows)
            for name, rows in self.ratio_rows.items()
        }
        self.statistics = {
            name: self._statistics(rows, self.ratios[name])
            for name, rows in self.ratio_rows.items()
        }
        self._print_statistics()

    def _relative_norm(self, rows: slice) -> np.ndarray:
        delta = self.delta_k[:, rows, :]
        baseline = self.Keff[:, rows, :]
        numerator = np.linalg.norm(delta, axis=(-2, -1))
        denominator = np.linalg.norm(baseline, axis=(-2, -1))
        valid = self.valid & (denominator > np.finfo(denominator.dtype).eps)
        ratio = np.full(len(self.steps), np.nan, dtype=np.float64)
        ratio[valid] = numerator[valid] / denominator[valid]
        return ratio

    def _statistics(self, rows: slice, ratio: np.ndarray) -> dict[str, float | int]:
        valid = np.isfinite(ratio)
        delta = self.delta_k[valid, rows, :]
        baseline = self.Keff[valid, rows, :]
        values = ratio[valid]
        return {
            "overall": float(np.linalg.norm(delta) / np.linalg.norm(baseline)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "valid_steps": int(valid.sum()),
        }

    def _print_statistics(self) -> None:
        print("[delta-k/k] group   overall      mean    median       p95       max  valid_steps")
        for name, stats in self.statistics.items():
            print(
                f"[delta-k/k] {name:<6} "
                f"{stats['overall']:9.5f} {stats['mean']:9.5f} "
                f"{stats['median']:9.5f} {stats['p95']:9.5f} "
                f"{stats['max']:9.5f} {stats['valid_steps']:12d}"
            )

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(19, 9.5))
        grid = self.fig.add_gridspec(
            3,
            4,
            left=0.03,
            right=0.98,
            bottom=0.13,
            top=0.90,
            wspace=0.32,
            hspace=0.72,
            width_ratios=(1.5, 1.5, 1.0, 1.0),
        )
        video_ax = self.fig.add_subplot(grid[:, :2])
        delta_ax = self.fig.add_subplot(grid[0, 2:])
        ratio_ax = self.fig.add_subplot(grid[1, 2:])
        phase_ax = self.fig.add_subplot(grid[2, 2:])

        self.video_image = video_ax.imshow(self._read_frame(0))
        video_ax.set_title("video")
        video_ax.axis("off")

        delta_limit = float(np.nanmax(self.delta_k[self.valid]))
        if delta_limit == 0.0:
            delta_limit = 1.0
        self.delta_limit = delta_limit
        self.delta_image = delta_ax.imshow(
            np.zeros_like(self.delta_k[0]),
            cmap="magma",
            aspect="auto",
            vmin=0.0,
            vmax=delta_limit,
        )
        self.fig.colorbar(self.delta_image, ax=delta_ax)
        configure_joint_matrix_axis(delta_ax, self.joint_names)
        self.delta_ax = delta_ax
        self.delta_value_texts = [
            delta_ax.text(col, row, "", ha="center", va="center", fontsize=5, visible=False)
            for row in range(self.delta_k.shape[-2])
            for col in range(self.delta_k.shape[-1])
        ]

        self.ratio_cursor = ratio_ax.axvline(0.0, color="gray", linestyle="--")
        self.ratio_markers = {}
        for name, ratio in self.ratios.items():
            color = self._colors[name]
            ratio_ax.plot(
                self.times,
                ratio,
                color=color,
                linewidth=2.0 if name == "Global" else 1.2,
                label=name,
            )
            (self.ratio_markers[name],) = ratio_ax.plot([], [], "o", color=color)
        ratio_ax.set_ylabel(r"$\|\Delta K\|_F / \|K\|_F$")
        ratio_ax.set_title("global and action-leg stiffness ratios")
        ratio_ax.grid(alpha=0.25)
        ratio_ax.legend(loc="best", ncols=5)

        phase_end = float(self.times[-1] + self.control_dt)
        phase_ax.imshow(
            self.foot_contact.T,
            cmap=ListedColormap(("white", "tab:blue")),
            interpolation="nearest",
            aspect="auto",
            vmin=0,
            vmax=1,
            extent=(float(self.times[0]), phase_end, 3.5, -0.5),
        )
        phase_ax.set_yticks(np.arange(len(self.foot_names)), self.foot_names)
        phase_ax.set_xlabel("time (s)")
        phase_ax.set_title("foot phase | white: swing, blue: contact")
        self.phase_cursor = phase_ax.axvline(0.0, color="red", linestyle="--")

        self._add_playback_controls()

    def _update_video_step(self, video_step: int) -> None:
        video_step = int(np.clip(video_step, 0, self.frame_count - 1))
        self.current_video_step = video_step
        self.video_image.set_data(self._read_frame(video_step))

        matrix_idx = min(video_step, len(self.steps) - 1)
        matrix_step = int(self.steps[matrix_idx])
        matrix_time = float(self.times[matrix_idx])
        self.delta_image.set_data(self.delta_k[matrix_idx])
        self.delta_ax.set_title(
            f"|Keff_aug - Keff| | step={matrix_step} | "
            f"Jdot_valid={bool(self.jdot_valid[matrix_idx])}"
        )
        self.ratio_cursor.set_xdata([matrix_time, matrix_time])
        self.phase_cursor.set_xdata([matrix_time, matrix_time])

        current_ratios = []
        for name, ratio in self.ratios.items():
            value = ratio[matrix_idx]
            if np.isfinite(value):
                self.ratio_markers[name].set_data([matrix_time], [value])
                current_ratios.append(f"{name}={value:.4f}")
            else:
                self.ratio_markers[name].set_data([], [])
                current_ratios.append(f"{name}=invalid")
        self.fig.suptitle(
            f"video step={video_step} | matrix step={matrix_step} | "
            f"t={matrix_time:.3f} s\n" + " | ".join(current_ratios)
        )
        self._update_value_annotations(matrix_idx)

        if int(self.slider.val) != video_step:
            self._updating_slider = True
            self.slider.set_val(video_step)
            self._updating_slider = False
        self.fig.canvas.draw_idle()

    def _update_value_annotations(self, matrix_idx: int) -> None:
        values = self.delta_k[matrix_idx].reshape(-1) if self.show_values else ()
        for idx, text in enumerate(self.delta_value_texts):
            text.set_visible(self.show_values)
            if self.show_values:
                value = float(values[idx])
                text.set_text(f"{value:.2g}")
                text.set_color("white" if value > 0.55 * self.delta_limit else "black")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Recorded play video path.")
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Recorded eff_impedance_timeseries.npz path.",
    )
    args = parser.parse_args()

    DeltaKViewer(video_path=args.video, npz_path=args.npz).show()


if __name__ == "__main__":
    main()
