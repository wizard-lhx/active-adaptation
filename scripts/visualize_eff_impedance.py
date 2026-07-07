"""Save recorded video with effective impedance heatmaps."""

from __future__ import annotations

import argparse
import itertools
import zipfile
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _matrix_index_for_video_step(steps: np.ndarray, video_step: int) -> int | None:
    idx = int(np.searchsorted(steps, video_step, side="right") - 1)
    if idx < 0:
        return None
    return min(idx, len(steps) - 1)


def save_replay(video_path: Path, npz_path: Path, output_path: Path | None, fps: float) -> Path:
    try:
        data = np.load(npz_path)
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            f"Could not read NPZ file {npz_path}: file is incomplete or corrupted. "
            "Regenerate it by running play again."
        ) from exc
    steps = data["steps"]
    keff = data["Keff"]
    deff = data["Deff"]
    output_path = output_path or video_path.with_name(f"{video_path.stem}_eff_impedance{video_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = iio.imiter(video_path)
    first_frame = next(frames)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.12))
    video_image = axes[0].imshow(first_frame)
    axes[0].set_title("video")
    axes[0].axis("off")

    keff_empty = np.zeros_like(keff[0])
    deff_empty = np.zeros_like(deff[0])
    keff_image = axes[1].imshow(
        keff_empty,
        cmap="coolwarm",
        aspect="auto",
        vmin=float(np.nanmin(keff)),
        vmax=float(np.nanmax(keff)),
    )
    deff_image = axes[2].imshow(
        deff_empty,
        cmap="coolwarm",
        aspect="auto",
        vmin=float(np.nanmin(deff)),
        vmax=float(np.nanmax(deff)),
    )
    axes[1].set_title("Keff")
    axes[2].set_title("Deff")
    fig.colorbar(keff_image, ax=axes[1])
    fig.colorbar(deff_image, ax=axes[2])
    for axis in axes[1:]:
        axis.set_xlabel("state joint")
        axis.set_ylabel("action joint")
    fig.tight_layout()

    writer = iio.imopen(output_path, "w").legacy_get_writer(fps=fps, codec="h264")
    try:
        current_matrix_idx = None
        for frame_idx, frame in enumerate(itertools.chain([first_frame], frames)):
            video_step = frame_idx + 1
            matrix_idx = _matrix_index_for_video_step(steps, video_step)
            video_image.set_data(frame)
            if matrix_idx is None:
                keff_image.set_data(keff_empty)
                deff_image.set_data(deff_empty)
                current_matrix_idx = None
            elif matrix_idx != current_matrix_idx:
                keff_image.set_data(keff[matrix_idx])
                deff_image.set_data(deff[matrix_idx])
                current_matrix_idx = matrix_idx
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    finally:
        writer.close()
        plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Recorded play video path.")
    parser.add_argument("--npz", type=Path, required=True, help="Recorded eff_impedance_timeseries.npz path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to '<input>_eff_impedance.mp4' next to the input video.",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Output video frame rate.")
    args = parser.parse_args()

    output_path = save_replay(
        video_path=args.video,
        npz_path=args.npz,
        output_path=args.output,
        fps=args.fps,
    )
    print(f"Saved effective impedance replay video to: {output_path}")


if __name__ == "__main__":
    main()
