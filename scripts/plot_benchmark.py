"""Chart the latency of each deployed version, from outputs/benchmark/benchmark.json.

Usage:
    python scripts/plot_benchmark.py

Shows the single-wafer case: batch size 1 on one thread, the simplest setting to compare.
"""

import json

import matplotlib.pyplot as plt

from waferwise.utils import repo_path

SURFACE, TEXT, MUTED, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
LABELS = {
    ("pytorch", "fp32"): "PyTorch FP32",
    ("openvino_python", "fp32"): "OpenVINO FP32, Python",
    ("openvino_cpp", "fp32"): "OpenVINO FP32, C++",
    ("openvino_python", "int8"): "OpenVINO INT8, Python",
    ("openvino_cpp", "int8"): "OpenVINO INT8, C++",
}


def main() -> None:
    with open(repo_path("outputs/benchmark/benchmark.json"), encoding="utf-8") as f:
        results = json.load(f)["results"]
    latency = {
        (r["runtime"], r["precision"]): r["latency_p50_ms"]["median"]
        for r in results if r["batch"] == 1 and r["threads"] == 1
    }
    keys = sorted(LABELS, key=latency.get, reverse=True)

    figure, axes = plt.subplots(figsize=(7.5, 2.8), dpi=150, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    positions = range(len(keys))
    axes.barh(positions, [latency[k] for k in keys], height=0.5, color=BAR)
    axes.set_yticks(positions, [LABELS[k] for k in keys], color=TEXT)
    for position, key in zip(positions, keys):
        axes.text(latency[key] + 0.04, position, f"{latency[key]:.2f} ms", va="center", color=TEXT)
    axes.set_title("Median time to classify one wafer, one CPU thread", loc="left", color=TEXT, fontsize=11)
    axes.tick_params(left=False, bottom=False, labelbottom=False)
    axes.spines[:].set_visible(False)
    axes.set_xlim(0, max(latency.values()) * 1.15)
    figure.tight_layout()
    path = repo_path("docs/figures/benchmark_latency.png")
    figure.savefig(path, facecolor=SURFACE)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
