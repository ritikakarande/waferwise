"""Exploratory analysis of the processed WM-811K data. Figures and tables go to outputs/eda/.

Usage:
    python scripts/eda.py [--config configs/data.yaml]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from waferwise.data import load_processed, load_splits
from waferwise.utils import get_logger, load_config, repo_path, save_json

log = get_logger("eda")
OUT_DIR = repo_path("outputs/eda")

# 0 = off-wafer, 1 = pass, 2 = fail
WAFER_CMAP = ListedColormap(["#f2f2f2", "#9db4c8", "#c0392b"])
BAR_COLOR = "#2c6e9e"


def class_counts(y, classes, splits) -> pd.DataFrame:
    table = pd.DataFrame({"class": classes, "total": np.bincount(y, minlength=len(classes))})
    for name, idx in splits.items():
        table[name] = np.bincount(y[idx], minlength=len(classes))
    table["share_%"] = (100 * table["total"] / table["total"].sum()).round(2)
    table["imbalance_vs_none"] = (table["total"].max() / table["total"]).round(1)
    return table.sort_values("total", ascending=False).reset_index(drop=True)


def plot_class_counts(table: pd.DataFrame) -> None:
    table = table.sort_values("total")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(table["class"], table["total"], color=BAR_COLOR, height=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("Number of wafers (log scale)")
    ax.set_title("WM-811K labelled wafers per class", loc="left")
    for i, n in enumerate(table["total"]):
        ax.text(n * 1.08, i, f"{n:,}", va="center", fontsize=9, color="#333333")
    ax.set_xlim(right=table["total"].max() * 4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_counts.png", dpi=150)
    plt.close(fig)


def plot_examples(X, y, classes, per_class: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.1, len(classes) * 1.15))
    for c, name in enumerate(classes):
        idx = rng.choice(np.flatnonzero(y == c), size=per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(X[i], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[c, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Random examples per class (64x64; grey = pass, red = fail)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "examples_per_class.png", dpi=130)
    plt.close(fig)


def plot_sizes(orig_shape: np.ndarray) -> None:
    h, w = orig_shape[:, 0], orig_shape[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(h * w, bins=80, color=BAR_COLOR)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Original map size (height x width, pixels)")
    axes[0].set_ylabel("Wafers (log scale)")
    axes[0].set_title("Map area", loc="left")
    axes[1].scatter(w, h, s=4, alpha=0.15, color=BAR_COLOR, linewidths=0)
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")
    axes[1].set_title("Height vs width (each dot = one wafer)", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "original_sizes.png", dpi=150)
    plt.close(fig)


def plot_fail_fraction(X, y, classes) -> np.ndarray:
    on_wafer = (X > 0).sum(axis=(1, 2))
    fail_frac = (X == 2).sum(axis=(1, 2)) / np.maximum(on_wafer, 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        [fail_frac[y == c] for c in range(len(classes))],
        tick_labels=classes,
        showfliers=False,
        medianprops={"color": "#c0392b"},
    )
    ax.set_ylabel("Fraction of dies that fail")
    ax.set_title("Failing-die fraction per class (outliers hidden)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fail_fraction_by_class.png", dpi=150)
    plt.close(fig)
    return fail_frac


def duplicate_report(hashes, y, classes, split_of) -> dict:
    """Exact duplicate wafer maps (same original pixels) and label conflicts among them."""
    df = pd.DataFrame({"hash": hashes, "y": y, "split": split_of})
    dup = df[df.duplicated("hash", keep=False)]
    groups = dup.groupby("hash")
    conflicting = groups["y"].nunique() > 1
    cross_split = groups["split"].nunique() > 1
    conflict_pairs = (
        dup[dup["hash"].isin(conflicting[conflicting].index)]
        .groupby("hash")["y"]
        .apply(lambda s: " / ".join(sorted({classes[v] for v in s})))
        .value_counts()
    )
    return {
        "duplicate_groups": int(groups.ngroups),
        "wafers_in_duplicate_groups": int(len(dup)),
        "groups_with_conflicting_labels": int(conflicting.sum()),
        "groups_spanning_multiple_splits": int(cross_split.sum()),
        "duplicates_by_class": {classes[c]: int(n) for c, n in dup["y"].value_counts().items()},
        "conflict_label_combinations": {k: int(v) for k, v in conflict_pairs.head(15).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_processed(cfg["processed_path"])
    X, y, classes = data["X"], data["y"], [str(c) for c in data["classes"]]
    splits = load_splits(cfg["splits_path"], data["source_index"])
    split_of = np.empty(len(y), dtype=object)
    for name, idx in splits.items():
        split_of[idx] = name

    table = class_counts(y, classes, splits)
    table.to_csv(OUT_DIR / "class_counts.csv", index=False)
    plot_class_counts(table)
    plot_examples(X, y, classes)
    plot_sizes(data["orig_shape"])
    fail_frac = plot_fail_fraction(X, y, classes)

    h, w = data["orig_shape"][:, 0], data["orig_shape"][:, 1]
    sizes = pd.Series([f"{a}x{b}" for a, b in zip(h, w)])
    none_idx = classes.index("none")
    defect = y != none_idx
    lots = pd.DataFrame({"lot": data["lot_name"], "split": split_of})

    summary = {
        "n_labelled": int(len(y)),
        "class_counts": dict(zip(table["class"], table["total"].astype(int).tolist())),
        "largest_to_smallest_class_ratio": float(table["total"].max() / table["total"].min()),
        "original_sizes": {
            "unique_shapes": int(sizes.nunique()),
            "height_min_max": [int(h.min()), int(h.max())],
            "width_min_max": [int(w.min()), int(w.max())],
            "most_common": {k: int(v) for k, v in sizes.value_counts().head(5).items()},
            "maps_with_side_below_20px": int(((h < 20) | (w < 20)).sum()),
            "maps_with_aspect_ratio_above_1.5": int((np.maximum(h, w) / np.minimum(h, w) > 1.5).sum()),
        },
        "duplicates": duplicate_report(data["map_hash"], y, classes, split_of),
        "fail_fraction": {
            "median_by_class": {c: round(float(np.median(fail_frac[y == i])), 4) for i, c in enumerate(classes)},
            "defect_labelled_with_no_failing_dies": int((defect & (fail_frac == 0)).sum()),
            "none_labelled_with_over_30pct_failing": int((~defect & (fail_frac > 0.3)).sum()),
        },
        "lots": {
            "n_lots": int(lots["lot"].nunique()),
            "lots_spanning_multiple_splits": int((lots.groupby("lot")["split"].nunique() > 1).sum()),
        },
        "original_train_test_label": {k: int(v) for k, v in pd.Series(data["orig_split"]).value_counts().items()},
    }
    save_json(summary, OUT_DIR / "summary.json")

    print(table.to_string(index=False))
    print()
    for key in ("original_sizes", "duplicates", "fail_fraction", "lots", "original_train_test_label"):
        print(f"{key}: {summary[key]}")
    log.info("Figures and tables written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
