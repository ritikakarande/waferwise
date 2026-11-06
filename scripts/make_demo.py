"""Build the demo files: the sample wafers, a copy of the INT8 model and the README animation.

Usage:
    python scripts/make_demo.py

The wafers are a seeded random draw from the validation split: one per class, plus three of the
wafers the gate sends to review. Nothing is picked by hand. Results shown in the animation come
from scripts/demo.py, so they use the committed model and the committed PNG files.
"""

import io
import json
import shutil

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from demo import classify  # noqa: E402
from waferwise.deploy import PIXEL_FOR_CODE, predict_openvino  # noqa: E402
from waferwise.train import load_data  # noqa: E402
from waferwise.utils import load_config, repo_path  # noqa: E402

SEED = 7
EXTRA_REVIEW_WAFERS = 3
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
TRACK, PASS_DIE, FAIL_DIE = "#e4e3df", "#c8c7c1", "#eb6834"
GOOD, WARNING = "#0ca30c", "#fab219"
WAFER_COLOURS = ListedColormap([SURFACE, PASS_DIE, FAIL_DIE])


def pick_wafers(y: np.ndarray, needs_review: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = [rng.choice(np.flatnonzero(y == c)) for c in np.unique(y)]
    review = np.setdiff1d(np.flatnonzero(needs_review), picks)
    picks.extend(rng.choice(review, size=EXTRA_REVIEW_WAFERS, replace=False))
    return rng.permutation(picks)


def draw_frame(wafer: np.ndarray, row: dict, threshold: float) -> Image.Image:
    automatic = row["decision"] == "AUTO"
    figure = plt.figure(figsize=(6.4, 2.8), dpi=100, facecolor=SURFACE)
    image_axes = figure.add_axes((0.03, 0.06, 0.36, 0.88))
    image_axes.imshow(wafer, cmap=WAFER_COLOURS, vmin=0, vmax=2, interpolation="nearest")
    image_axes.axis("off")

    figure.text(0.44, 0.84, row["predicted"], fontsize=20, fontweight="bold", color=TEXT)
    figure.text(0.44, 0.75, f"model prediction, {row['confidence']:.1%} confident", fontsize=10, color=MUTED)

    bar = figure.add_axes((0.44, 0.60, 0.52, 0.07))
    bar.barh(0, 1, color=TRACK, height=1)
    bar.barh(0, row["confidence"], color=TEXT, height=1)
    bar.axvline(threshold, color=SURFACE, linewidth=3)
    bar.set(xlim=(0, 1), ylim=(-0.5, 0.5))
    bar.axis("off")
    figure.text(0.44 + 0.52 * threshold, 0.53, f"gate {threshold:.2f}", fontsize=9, color=MUTED, ha="center")

    chip = "✓ AUTO" if automatic else "⚠ REVIEW"
    figure.text(
        0.44, 0.31, chip, fontsize=14, fontweight="bold", color=TEXT,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": GOOD if automatic else WARNING, "edgecolor": "none"},
    )
    agrees = "agrees" if row["label"] == row["predicted"] else "differs"
    figure.text(0.44, 0.12, f"dataset label: {row['label']} ({agrees})", fontsize=10, color=MUTED)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=SURFACE)
    plt.close(figure)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    deploy_dir = repo_path(load_config("configs/deploy.yaml")["out_dir"])
    model_dir = repo_path("models")
    model_dir.mkdir(exist_ok=True)
    for name in ("wafer_cnn_int8.xml", "wafer_cnn_int8.bin", "gate.json"):
        shutil.copy(deploy_dir / name, model_dir / name)
    with open(model_dir / "gate.json", encoding="utf-8") as f:
        gate = json.load(f)

    data, splits, classes = load_data("configs/data.yaml")
    X, y = data["X"][splits["val"]], data["y"][splits["val"]]
    confidence = predict_openvino(model_dir / "wafer_cnn_int8.xml", X).max(axis=1)
    picks = pick_wafers(y, confidence < gate["threshold"], SEED)

    wafer_dir = repo_path("demo/wafers")
    shutil.rmtree(wafer_dir, ignore_errors=True)
    wafer_dir.mkdir(parents=True)
    paths = [wafer_dir / f"{n:02d}_{classes[y[i]]}.png" for n, i in enumerate(picks, start=1)]
    for path, i in zip(paths, picks):
        cv2.imwrite(str(path), PIXEL_FOR_CODE[X[i]])

    rows = classify(model_dir / "wafer_cnn_int8.xml", gate, paths)
    frames = [draw_frame(X[i], row, gate["threshold"]) for i, row in zip(picks, rows)]
    gif_path = repo_path("docs/figures/demo.gif")
    all_frames = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for n, frame in enumerate(frames):
        all_frames.paste(frame, (0, n * frame.height))
    palette = all_frames.quantize(colors=48)  # shared by all frames, so it needs every frame's colours
    frames[0].quantize(palette=palette).save(
        gif_path, save_all=True, loop=0, duration=1600, optimize=True,
        append_images=[frame.quantize(palette=palette) for frame in frames[1:]],
    )
    print(f"Wrote {len(paths)} wafers and {gif_path}")


if __name__ == "__main__":
    main()
