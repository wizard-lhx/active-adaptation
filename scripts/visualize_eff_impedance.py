"""Interactively replay video with recorded effective impedance diagnostics."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class Viewer:
    """Matplotlib viewer with slider and held left/right navigation."""

    def __init__(self, video_path: Path, npz_path: Path) -> None:
        with np.load(npz_path) as data:
            self.steps = data["steps"]
            self.Keff = data["Keff"]
            self.Deff = data["Deff"]
            self.Keff_eigvals = data["Keff_eigvals"]
            self.Deff_eigvals = data["Deff_eigvals"]
            self.Keff_cond = data["Keff_cond"]
            self.Deff_cond = data["Deff_cond"]
            self.joint_names = data["joint_names"].astype(str)
            self.control_dt = float(data["control_dt"])

        self.times = self.steps * self.control_dt
        self.Keff_min = self.Keff_eigvals[:, 0]
        self.Deff_min = self.Deff_eigvals[:, 0]
        self.Keff_neg_count = (self.Keff_eigvals < 0.0).sum(axis=-1)
        self.Deff_neg_count = (self.Deff_eigvals < 0.0).sum(axis=-1)

        video_meta = iio.immeta(video_path)
        self.video_fps = float(video_meta["fps"])
        self.frame_count = int(round(float(video_meta["duration"]) * float(video_meta["fps"])))
        self.video_reader = iio.imopen(video_path, "r")
        self.current_video_step = 0
        self.pressed_keys: set[str] = set()
        self.is_playing = False
        self.show_values = False
        self._updating_slider = False

        self._build_figure()
        self.key_timer = self.fig.canvas.new_timer(interval=50)
        self.key_timer.add_callback(self._repeat_held_key)
        self.play_timer = self.fig.canvas.new_timer(interval=int(round(1000.0 / self.video_fps)))
        self.play_timer.add_callback(self._advance_playback)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self._update_video_step(0)

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
            joint_ids = np.arange(len(self.joint_names))
            axis.set_xticks(joint_ids, self.joint_names, rotation=90, fontsize=6)
            axis.set_yticks(joint_ids, self.joint_names, fontsize=6)
        self.keff_ax = keff_ax
        self.deff_ax = deff_ax
        self.matrix_value_texts = {
            "Keff": [
                keff_ax.text(col, row, "", ha="center", va="center", fontsize=5, visible=False)
                for row in range(self.Keff.shape[-2])
                for col in range(self.Keff.shape[-1])
            ],
            "Deff": [
                deff_ax.text(col, row, "", ha="center", va="center", fontsize=5, visible=False)
                for row in range(self.Deff.shape[-2])
                for col in range(self.Deff.shape[-1])
            ],
        }
        self.matrix_value_limits = {"Keff": keff_limit, "Deff": deff_limit}

        eig_ids = np.arange(self.Keff_eigvals.shape[-1])
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
        eig_min = min(float(np.nanmin(self.Keff_eigvals)), float(np.nanmin(self.Deff_eigvals)), 0.0)
        eig_max = max(float(np.nanmax(self.Keff_eigvals)), float(np.nanmax(self.Deff_eigvals)), 0.0)
        eig_pad = 0.05 * (eig_max - eig_min)
        eig_ax.set_xlim(eig_ids[0] - 0.5, eig_ids[-1] + 0.5)
        eig_ax.set_ylim(eig_min - eig_pad, eig_max + eig_pad)
        eig_ax.set_xticks(eig_ids)
        eig_ax.axhline(0.0, color="black", linewidth=1.0)
        eig_ax.set_xlabel("eigenvalue index")
        eig_ax.set_ylabel("symmetric eigenvalue")
        eig_ax.set_title("current eigenvalue spectrum")
        eig_ax.legend()
        self.keff_eig_texts = [
            eig_ax.text(idx, 0.0, "", ha="center", va="bottom", fontsize=7, visible=False)
            for idx in eig_ids
        ]
        self.deff_eig_texts = [
            eig_ax.text(idx, 0.0, "", ha="center", va="top", fontsize=7, visible=False)
            for idx in eig_ids
        ]

        min_eig_ax.plot(self.times, self.Keff_min, label="Keff")
        min_eig_ax.plot(self.times, self.Deff_min, label="Deff")
        min_eig_ax.axhline(0.0, color="black", linewidth=1.0)
        self.min_eig_cursor = min_eig_ax.axvline(0, color="black", linestyle="--")
        (self.keff_min_marker,) = min_eig_ax.plot([], [], "o")
        (self.deff_min_marker,) = min_eig_ax.plot([], [], "o")
        min_eig_ax.set_ylabel("min eigenvalue")
        min_eig_ax.legend(loc="best")

        cond_ax.plot(self.times, self.Keff_cond, label="Keff")
        cond_ax.plot(self.times, self.Deff_cond, label="Deff")
        self.cond_cursor = cond_ax.axvline(0, color="black", linestyle="--")
        (self.keff_cond_marker,) = cond_ax.plot([], [], "o")
        (self.deff_cond_marker,) = cond_ax.plot([], [], "o")
        cond_ax.set_yscale("log")
        cond_ax.set_xlabel("time (s)")
        cond_ax.set_ylabel("condition number")
        cond_ax.legend(loc="best")

        slider_ax = self.fig.add_axes((0.10, 0.045, 0.80, 0.03))
        self.slider = Slider(
            slider_ax,
            "video step",
            0,
            self.frame_count - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider)
        self.controls_text = self.fig.text(
            0.5,
            0.095,
            "PAUSED | SPACE: play/pause | hold LEFT/RIGHT: play steps | V: values off",
            ha="center",
        )

    def _update_video_step(self, video_step: int) -> None:
        video_step = int(np.clip(video_step, 0, self.frame_count - 1))
        self.current_video_step = video_step
        self.video_image.set_data(self._read_frame(video_step))

        matrix_idx = min(video_step, len(self.steps) - 1)
        matrix_step = int(self.steps[matrix_idx])
        matrix_time = float(self.times[matrix_idx])
        self.min_eig_cursor.set_xdata([matrix_time, matrix_time])
        self.cond_cursor.set_xdata([matrix_time, matrix_time])
        self.keff_image.set_data(self.Keff[matrix_idx])
        self.deff_image.set_data(self.Deff[matrix_idx])
        self.keff_eig_line.set_ydata(self.Keff_eigvals[matrix_idx])
        self.deff_eig_line.set_ydata(self.Deff_eigvals[matrix_idx])
        self.keff_min_marker.set_data([matrix_time], [self.Keff_min[matrix_idx]])
        self.deff_min_marker.set_data([matrix_time], [self.Deff_min[matrix_idx]])
        self.keff_cond_marker.set_data([matrix_time], [self.Keff_cond[matrix_idx]])
        self.deff_cond_marker.set_data([matrix_time], [self.Deff_cond[matrix_idx]])
        self.keff_ax.set_title(f"Keff | step={matrix_step} | blue < 0 < red")
        self.deff_ax.set_title(f"Deff | step={matrix_step} | blue < 0 < red")
        joints = len(self.joint_names)
        self.fig.suptitle(
            f"video step={video_step} | matrix step={matrix_step} | t={matrix_time:.3f} s\n"
            f"K: min eig={self.Keff_min[matrix_idx]:.5g}, cond={self.Keff_cond[matrix_idx]:.5g}, "
            f"negative={int(self.Keff_neg_count[matrix_idx])} ({self.Keff_neg_count[matrix_idx] / joints:.1%}) | "
            f"D: min eig={self.Deff_min[matrix_idx]:.5g}, cond={self.Deff_cond[matrix_idx]:.5g}, "
            f"negative={int(self.Deff_neg_count[matrix_idx])} ({self.Deff_neg_count[matrix_idx] / joints:.1%})"
        )
        self._update_value_annotations(matrix_idx)

        if int(self.slider.val) != video_step:
            self._updating_slider = True
            self.slider.set_val(video_step)
            self._updating_slider = False
        self.fig.canvas.draw_idle()

    def _on_slider(self, value: float) -> None:
        if not self._updating_slider:
            self._set_playing(False)
            self._update_video_step(int(value))

    def _on_key_press(self, event) -> None:
        key = event.key
        if key == " ":
            self.pressed_keys.difference_update({"left", "right"})
            self.key_timer.stop()
            self._set_playing(not self.is_playing)
            return
        if key == "v":
            self.show_values = not self.show_values
            matrix_idx = min(self.current_video_step, len(self.steps) - 1)
            self._update_value_annotations(matrix_idx)
            self._update_controls_text()
            self.fig.canvas.draw_idle()
            return
        if key not in {"left", "right"} or key in self.pressed_keys:
            return
        self.pressed_keys.add(key)
        self._set_playing(False)
        self._repeat_held_key()
        self.key_timer.start()

    def _on_key_release(self, event) -> None:
        self.pressed_keys.discard(event.key)
        if not self.pressed_keys.intersection({"left", "right"}):
            self.key_timer.stop()

    def _repeat_held_key(self) -> None:
        if "right" in self.pressed_keys:
            self._update_video_step(self.current_video_step + 1)
        elif "left" in self.pressed_keys:
            self._update_video_step(self.current_video_step - 1)

    def _advance_playback(self) -> None:
        if self.current_video_step >= self.frame_count - 1:
            self._set_playing(False)
            return
        self._update_video_step(self.current_video_step + 1)
        if self.current_video_step >= self.frame_count - 1:
            self._set_playing(False)

    def _set_playing(self, playing: bool) -> None:
        if self.is_playing == playing:
            return
        self.is_playing = playing
        if playing:
            self.play_timer.start()
        else:
            self.play_timer.stop()
        self._update_controls_text()
        self.fig.canvas.draw_idle()

    def _update_controls_text(self) -> None:
        playback = "PLAYING" if self.is_playing else "PAUSED"
        values = "on" if self.show_values else "off"
        self.controls_text.set_text(
            f"{playback} | SPACE: play/pause | hold LEFT/RIGHT: play steps | V: values {values}"
        )

    def _update_value_annotations(self, matrix_idx: int) -> None:
        visible = self.show_values
        matrices = {"Keff": self.Keff, "Deff": self.Deff}
        for key, texts in self.matrix_value_texts.items():
            values = matrices[key][matrix_idx].reshape(-1) if visible else ()
            limit = self.matrix_value_limits[key]
            for idx, text in enumerate(texts):
                text.set_visible(visible)
                if visible:
                    value = float(values[idx])
                    text.set_text(f"{value:.2g}")
                    text.set_color("white" if abs(value) > 0.55 * limit else "black")

        eig_groups = (
            (self.keff_eig_texts, self.Keff_eigvals),
            (self.deff_eig_texts, self.Deff_eigvals),
        )
        for texts, eigvals in eig_groups:
            values = eigvals[matrix_idx] if visible else ()
            for idx, text in enumerate(texts):
                text.set_visible(visible)
                if visible:
                    value = float(values[idx])
                    text.set_position((idx, value))
                    text.set_text(f"{value:.2g}")

    def show(self) -> None:
        try:
            plt.show()
        finally:
            self.play_timer.stop()
            self.key_timer.stop()
            self._read_frame.cache_clear()
            self.video_reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Recorded play video path.")
    parser.add_argument("--npz", type=Path, required=True, help="Recorded eff_impedance_timeseries.npz path.")
    args = parser.parse_args()

    Viewer(video_path=args.video, npz_path=args.npz).show()


if __name__ == "__main__":
    main()
