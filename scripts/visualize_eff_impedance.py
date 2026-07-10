"""Interactively replay video with recorded effective impedance diagnostics."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def _matrix_index_for_video_step(steps: np.ndarray, video_step: int) -> int | None:
    idx = int(np.searchsorted(steps, video_step, side="right") - 1)
    if idx < 0:
        return None
    return min(idx, len(steps) - 1)


class EffImpedanceReplayViewer:
    """Matplotlib viewer with slider and held left/right navigation."""

    def __init__(self, video_path: Path, npz_path: Path) -> None:
        with np.load(npz_path) as data:
            self.steps = data["steps"]
            self.num_points = data["num_points"]
            self.Keff = data["Keff"]
            self.Deff = data["Deff"]
            self.Keff_sym_eigvals = data["Keff_sym_eigvals"]
            self.Deff_sym_eigvals = data["Deff_sym_eigvals"]
            self.Keff_sym_min_eig = data["Keff_sym_min_eig"]
            self.Deff_sym_min_eig = data["Deff_sym_min_eig"]
            self.Keff_sym_cond = data["Keff_sym_cond"]
            self.Deff_sym_cond = data["Deff_sym_cond"]
            self.Keff_sym_neg_count = data["Keff_sym_neg_count"]
            self.Deff_sym_neg_count = data["Deff_sym_neg_count"]
            self.Keff_sym_neg_frac = data["Keff_sym_neg_frac"]
            self.Deff_sym_neg_frac = data["Deff_sym_neg_frac"]

        video_meta = iio.immeta(video_path)
        self.frame_count = int(round(float(video_meta["duration"]) * float(video_meta["fps"])))
        self.video_reader = iio.imopen(video_path, "r")
        self.current_video_step = 1
        self.held_keys: set[str] = set()
        self._updating_slider = False

        self._build_figure()
        self.key_timer = self.fig.canvas.new_timer(interval=50)
        self.key_timer.add_callback(self._repeat_held_key)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self._update_video_step(1)

    @lru_cache(maxsize=16)
    def _read_frame(self, frame_idx: int) -> np.ndarray:
        return self.video_reader.read(index=frame_idx)

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(19, 9.5))
        grid = self.fig.add_gridspec(
            2,
            4,
            left=0.03,
            right=0.98,
            bottom=0.13,
            top=0.90,
            wspace=0.32,
            hspace=0.32,
            width_ratios=(1.5, 1.5, 1.0, 1.0),
        )
        video_ax = self.fig.add_subplot(grid[:, :2])
        keff_ax = self.fig.add_subplot(grid[0, 2])
        deff_ax = self.fig.add_subplot(grid[0, 3])
        eig_ax = self.fig.add_subplot(grid[1, 2])
        trend_grid = grid[1, 3].subgridspec(2, 1, hspace=0.55)
        min_eig_ax = self.fig.add_subplot(trend_grid[0, 0])
        cond_ax = self.fig.add_subplot(trend_grid[1, 0])

        self.video_image = video_ax.imshow(self._read_frame(0))
        video_ax.set_title("video")
        video_ax.axis("off")

        keff_limit = float(np.nanmax(np.abs(self.Keff)))
        deff_limit = float(np.nanmax(np.abs(self.Deff)))
        self.keff_image = keff_ax.imshow(
            np.zeros_like(self.Keff[0]),
            cmap="coolwarm",
            aspect="auto",
            vmin=-keff_limit,
            vmax=keff_limit,
        )
        self.deff_image = deff_ax.imshow(
            np.zeros_like(self.Deff[0]),
            cmap="coolwarm",
            aspect="auto",
            vmin=-deff_limit,
            vmax=deff_limit,
        )
        self.fig.colorbar(self.keff_image, ax=keff_ax)
        self.fig.colorbar(self.deff_image, ax=deff_ax)
        for axis in (keff_ax, deff_ax):
            axis.set_xlabel("state joint")
            axis.set_ylabel("action joint")
        self.keff_ax = keff_ax
        self.deff_ax = deff_ax

        eig_ids = np.arange(self.Keff_sym_eigvals.shape[-1])
        (self.keff_eig_line,) = eig_ax.plot(
            eig_ids,
            np.full_like(eig_ids, np.nan, dtype=float),
            "o-",
            label="Keff",
        )
        (self.deff_eig_line,) = eig_ax.plot(
            eig_ids,
            np.full_like(eig_ids, np.nan, dtype=float),
            "o-",
            label="Deff",
        )
        eig_min = min(float(np.nanmin(self.Keff_sym_eigvals)), float(np.nanmin(self.Deff_sym_eigvals)), 0.0)
        eig_max = max(float(np.nanmax(self.Keff_sym_eigvals)), float(np.nanmax(self.Deff_sym_eigvals)), 0.0)
        eig_pad = 0.05 * (eig_max - eig_min)
        eig_ax.set_ylim(eig_min - eig_pad, eig_max + eig_pad)
        eig_ax.axhline(0.0, color="black", linewidth=1.0)
        eig_ax.set_xlabel("eigenvalue index")
        eig_ax.set_ylabel("symmetric eigenvalue")
        eig_ax.set_title("current eigenvalue spectrum")
        eig_ax.legend()

        min_eig_ax.plot(self.steps, self.Keff_sym_min_eig, label="Keff")
        min_eig_ax.plot(self.steps, self.Deff_sym_min_eig, label="Deff")
        min_eig_ax.axhline(0.0, color="black", linewidth=1.0)
        self.min_eig_cursor = min_eig_ax.axvline(1, color="black", linestyle="--")
        (self.keff_min_marker,) = min_eig_ax.plot([], [], "o")
        (self.deff_min_marker,) = min_eig_ax.plot([], [], "o")
        min_eig_ax.set_ylabel("min eigenvalue")
        min_eig_ax.legend(loc="best")

        cond_ax.plot(self.steps, self.Keff_sym_cond, label="Keff")
        cond_ax.plot(self.steps, self.Deff_sym_cond, label="Deff")
        self.cond_cursor = cond_ax.axvline(1, color="black", linestyle="--")
        (self.keff_cond_marker,) = cond_ax.plot([], [], "o")
        (self.deff_cond_marker,) = cond_ax.plot([], [], "o")
        cond_ax.set_yscale("log")
        cond_ax.set_xlabel("play step")
        cond_ax.set_ylabel("condition number")
        cond_ax.legend(loc="best")

        slider_ax = self.fig.add_axes((0.10, 0.045, 0.80, 0.03))
        self.slider = Slider(
            slider_ax,
            "video step",
            1,
            self.frame_count,
            valinit=1,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider)
        self.fig.text(
            0.5,
            0.095,
            "drag slider | hold left/right: play steps",
            ha="center",
        )

    def _update_video_step(self, video_step: int) -> None:
        video_step = int(np.clip(video_step, 1, self.frame_count))
        self.current_video_step = video_step
        self.video_image.set_data(self._read_frame(video_step - 1))
        self.min_eig_cursor.set_xdata([video_step, video_step])
        self.cond_cursor.set_xdata([video_step, video_step])

        matrix_idx = _matrix_index_for_video_step(self.steps, video_step)
        if matrix_idx is None:
            self.keff_image.set_data(np.zeros_like(self.Keff[0]))
            self.deff_image.set_data(np.zeros_like(self.Deff[0]))
            self.keff_eig_line.set_ydata(np.full(self.Keff.shape[-1], np.nan))
            self.deff_eig_line.set_ydata(np.full(self.Deff.shape[-1], np.nan))
            for marker in (
                self.keff_min_marker,
                self.deff_min_marker,
                self.keff_cond_marker,
                self.deff_cond_marker,
            ):
                marker.set_data([], [])
            self.keff_ax.set_title("Keff | blue < 0 < red")
            self.deff_ax.set_title("Deff | blue < 0 < red")
            self.fig.suptitle(f"video step={video_step}")
        else:
            matrix_step = int(self.steps[matrix_idx])
            self.keff_image.set_data(self.Keff[matrix_idx])
            self.deff_image.set_data(self.Deff[matrix_idx])
            self.keff_eig_line.set_ydata(self.Keff_sym_eigvals[matrix_idx])
            self.deff_eig_line.set_ydata(self.Deff_sym_eigvals[matrix_idx])
            self.keff_min_marker.set_data([matrix_step], [self.Keff_sym_min_eig[matrix_idx]])
            self.deff_min_marker.set_data([matrix_step], [self.Deff_sym_min_eig[matrix_idx]])
            self.keff_cond_marker.set_data([matrix_step], [self.Keff_sym_cond[matrix_idx]])
            self.deff_cond_marker.set_data([matrix_step], [self.Deff_sym_cond[matrix_idx]])
            self.keff_ax.set_title(f"Keff | step={matrix_step} | blue < 0 < red")
            self.deff_ax.set_title(f"Deff | step={matrix_step} | blue < 0 < red")
            self.fig.suptitle(
                f"video step={video_step} | matrix step={matrix_step} | n={int(self.num_points[matrix_idx])}\n"
                f"K: min eig={self.Keff_sym_min_eig[matrix_idx]:.5g}, "
                f"cond={self.Keff_sym_cond[matrix_idx]:.5g}, "
                f"negative={int(self.Keff_sym_neg_count[matrix_idx])} "
                f"({self.Keff_sym_neg_frac[matrix_idx]:.1%}) | "
                f"D: min eig={self.Deff_sym_min_eig[matrix_idx]:.5g}, "
                f"cond={self.Deff_sym_cond[matrix_idx]:.5g}, "
                f"negative={int(self.Deff_sym_neg_count[matrix_idx])} "
                f"({self.Deff_sym_neg_frac[matrix_idx]:.1%})"
            )

        if int(self.slider.val) != video_step:
            self._updating_slider = True
            self.slider.set_val(video_step)
            self._updating_slider = False
        self.fig.canvas.draw_idle()

    def _on_slider(self, value: float) -> None:
        if not self._updating_slider:
            self._update_video_step(int(value))

    def _on_key_press(self, event) -> None:
        key = event.key
        if key not in {"left", "right"} or key in self.held_keys:
            return
        self.held_keys.add(key)
        self._repeat_held_key()
        self.key_timer.start()

    def _on_key_release(self, event) -> None:
        self.held_keys.discard(event.key)
        if not self.held_keys:
            self.key_timer.stop()

    def _repeat_held_key(self) -> None:
        if "right" in self.held_keys:
            self._update_video_step(self.current_video_step + 1)
        elif "left" in self.held_keys:
            self._update_video_step(self.current_video_step - 1)

    def show(self) -> None:
        try:
            plt.show()
        finally:
            self.key_timer.stop()
            self._read_frame.cache_clear()
            self.video_reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Recorded play video path.")
    parser.add_argument("--npz", type=Path, required=True, help="Recorded eff_impedance_timeseries.npz path.")
    args = parser.parse_args()

    EffImpedanceReplayViewer(video_path=args.video, npz_path=args.npz).show()


if __name__ == "__main__":
    main()
