"""Generate synthetic wafer maps for the rare classes with the trained conditional VAE.

Usage:
    python scripts/generate_synthetic.py [--config configs/cvae.yaml]

Writes data/synthetic/cvae.npz (X, y, mask_source_index) and a real-vs-synthetic preview grid.
Synthetic wafers are only ever added to the TRAINING set.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from waferwise.generative import ConditionalVAE, sample_wafers
from waferwise.train import drop_listed_wafers, load_data
from waferwise.utils import get_logger, load_config, repo_path, set_seed

log = get_logger("generate_synthetic")
WAFER_CMAP = ListedColormap(["#f2f2f2", "#9db4c8", "#c0392b"])  # off-wafer, pass, fail


def plot_real_vs_synthetic(real: dict, synth: dict, path, per_side: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    names = list(synth)
    fig, axes = plt.subplots(len(names), 2 * per_side + 1, figsize=((2 * per_side + 1) * 1.05, len(names) * 1.2))
    for r, name in enumerate(names):
        picks = [real[name][i] for i in rng.choice(len(real[name]), per_side, replace=False)]
        picks += [None] + [synth[name][i] for i in rng.choice(len(synth[name]), per_side, replace=False)]
        for c, wafer in enumerate(picks):
            ax = axes[r, c]
            ax.axis("off")
            if wafer is not None:
                ax.imshow(wafer, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[r, 0].text(-8, 32, name, ha="right", va="center", fontsize=10)
    axes[0, per_side // 2].set_title("Real (training set)", fontsize=10)
    axes[0, per_side + 1 + per_side // 2].set_title("Synthetic (cVAE)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/cvae.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    gcfg = cfg["generate"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = repo_path(f"outputs/{cfg['experiment_name']}")

    checkpoint = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = ConditionalVAE(len(classes), **cfg["model"])
    model.load_state_dict(checkpoint["model_state"])

    data, splits, _ = load_data(cfg["data_config"])
    train_idx = drop_listed_wafers(splits["train"], data["source_index"], cfg["drop_list"])

    X_out, y_out, mask_src, real, synth = [], [], [], {}, {}
    for name in gcfg["classes"]:
        c = classes.index(name)
        class_idx = train_idx[data["y"][train_idx] == c]
        masks = data["X"][class_idx] > 0
        die_shapes = data["orig_shape"][class_idx]
        X, mask_ids = sample_wafers(model, c, masks, die_shapes, gcfg["per_class"], rng,
                                    gcfg["temperature"], gcfg.get("sharpen", 1.0))
        X_out.append(X)
        y_out.append(np.full(len(X), c, dtype=np.int64))
        mask_src.append(data["source_index"][class_idx[mask_ids]])
        real[name], synth[name] = data["X"][class_idx], X
        log.info("%s: generated %d (from %d real training wafers)", name, len(X), len(class_idx))

    out_path = repo_path(gcfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=np.concatenate(X_out), y=np.concatenate(y_out),
                        mask_source_index=np.concatenate(mask_src))
    plot_real_vs_synthetic(real, synth, out_dir / "real_vs_synthetic.png")
    log.info("Saved %s and %s", out_path, out_dir / "real_vs_synthetic.png")


if __name__ == "__main__":
    main()
