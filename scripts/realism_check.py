"""How realistic and useful are the synthetic wafers? Results go to outputs/realism/.

Checks, per rare class:
1. Real-vs-synthetic detectors (a small CNN and gradient-boosted trees on hand-made features), 5-fold
   cross-validated. ROC AUC 0.5 = indistinguishable from real, 1.0 = trivially spotted.
2. Recognisability: share of synthetic wafers the baseline classifier assigns to the intended class.
3. Copying: nearest-neighbour distance from each synthetic fail map to the real training maps of its
   class, compared with real-to-real nearest-neighbour distances.
4. Failing-die fraction, real vs synthetic.

Usage:
    python scripts/realism_check.py [--config configs/cvae.yaml] [--threads N]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn

from waferwise.data import WaferDataset
from waferwise.evaluate import predict
from waferwise.generative import fail_maps
from waferwise.label_audit import wafer_features
from waferwise.models import WaferCNN
from waferwise.train import drop_listed_wafers, load_checkpoint, load_data, random_dihedral
from waferwise.utils import get_logger, load_config, repo_path, save_json, set_seed

log = get_logger("realism_check")
OUT_DIR = repo_path("outputs/realism")


def cnn_detector_scores(X, is_synth, strata, seed, epochs=8) -> np.ndarray:
    """Out-of-fold P(synthetic) from a small CNN trained to tell real from synthetic."""
    scores = np.zeros(len(X), dtype=np.float32)
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
        torch.manual_seed(seed + fold)
        model = WaferCNN(num_classes=2, channels=(8, 16, 32, 64), dropout=0.2).to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        loader = torch.utils.data.DataLoader(
            WaferDataset(X, is_synth, tr, transform=random_dihedral), batch_size=128, shuffle=True,
            generator=torch.Generator().manual_seed(seed + fold),
        )
        for _ in range(epochs):
            model.train()
            for x, y in loader:
                optimizer.zero_grad()
                nn.functional.cross_entropy(model(x.contiguous(memory_format=torch.channels_last)), y).backward()
                optimizer.step()
        test_loader = torch.utils.data.DataLoader(WaferDataset(X, is_synth, te), batch_size=1024)
        scores[te] = predict(model, test_loader)[0][:, 1]
    return scores


def tree_detector_scores(X, is_synth, strata, seed) -> np.ndarray:
    feats = wafer_features(X)
    scores = np.zeros(len(X), dtype=np.float32)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
        model = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(feats[tr], is_synth[tr])
        scores[te] = model.predict_proba(feats[te])[:, 1]
    return scores


def nearest_distances(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """For each fail map in ``a``, the share of pixels that differ from its nearest map in ``b``."""
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    d = (a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * a @ b.T) / a.shape[1]
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(1)


def failing_fraction(maps: np.ndarray) -> np.ndarray:
    return (maps == 2).sum((1, 2)) / np.maximum((maps > 0).sum((1, 2)), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/cvae.yaml")
    parser.add_argument("--classifier", default="outputs/baseline", help="run folder of the classifier used for check 2")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data, splits, classes = load_data(cfg["data_config"])
    train_idx = drop_listed_wafers(splits["train"], data["source_index"], cfg["drop_list"])
    with np.load(repo_path(cfg["generate"]["output"])) as npz:
        synth_X, synth_y = npz["X"], npz["y"]

    # Balanced real-vs-synthetic set: for each class, all real training wafers + the same number of synthetic ones.
    rng = np.random.default_rng(cfg["seed"])
    parts_X, parts_cls, parts_synth = [], [], []
    for name in cfg["generate"]["classes"]:
        c = classes.index(name)
        real = data["X"][train_idx[data["y"][train_idx] == c]]
        fake = synth_X[synth_y == c]
        fake = fake[rng.choice(len(fake), size=min(len(real), len(fake)), replace=False)]
        parts_X += [real, fake]
        parts_cls += [np.full(len(real) + len(fake), c)]
        parts_synth += [np.zeros(len(real), dtype=np.int64), np.ones(len(fake), dtype=np.int64)]
    X = np.concatenate(parts_X)
    cls = np.concatenate(parts_cls)
    is_synth = np.concatenate(parts_synth)
    strata = cls * 2 + is_synth

    log.info("Training real-vs-synthetic detectors on %d wafers", len(X))
    detectors = {"cnn": cnn_detector_scores(X, is_synth, strata, cfg["seed"]),
                 "trees": tree_detector_scores(X, is_synth, strata, cfg["seed"])}

    classifier, _, _ = load_checkpoint(repo_path(args.classifier) / "model.pt")
    synth_pred = predict(classifier, torch.utils.data.DataLoader(WaferDataset(synth_X, synth_y), batch_size=1024))[0].argmax(1)

    results, fail_frac = {}, {}
    for name in cfg["generate"]["classes"]:
        c = classes.index(name)
        sel = cls == c
        real_fail = fail_maps(X[sel & (is_synth == 0)])
        fake_fail = fail_maps(synth_X[synth_y == c])
        real_nn = nearest_distances(real_fail, real_fail, exclude_self=True)
        fake_nn = nearest_distances(fake_fail, real_fail)
        fail_frac[name] = (failing_fraction(X[sel & (is_synth == 0)]), failing_fraction(synth_X[synth_y == c]))
        preds = synth_pred[synth_y == c]
        wrong = preds[preds != c]
        results[name] = {
            "real_train_wafers": int((sel & (is_synth == 0)).sum()),
            "detector_auc": {k: round(float(roc_auc_score(is_synth[sel], s[sel])), 3) for k, s in detectors.items()},
            "classified_as_intended_class": round(float((preds == c).mean()), 3),
            "most_common_wrong_prediction": classes[np.bincount(wrong).argmax()] if len(wrong) else None,
            "nn_distance_median": {"real_to_real": round(float(np.median(real_nn)), 4),
                                   "synthetic_to_real": round(float(np.median(fake_nn)), 4)},
            "synthetic_closer_than_5pct_of_real_pairs": round(float((fake_nn < np.percentile(real_nn, 5)).mean()), 3),
            "fail_fraction_median": {"real": round(float(np.median(fail_frac[name][0])), 3),
                                     "synthetic": round(float(np.median(fail_frac[name][1])), 3)},
        }
    summary = {
        "overall_detector_auc": {k: round(float(roc_auc_score(is_synth, s)), 3) for k, s in detectors.items()},
        "per_class": results,
    }
    save_json(summary, OUT_DIR / "summary.json")

    names = list(fail_frac)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pos = np.arange(len(names)) * 3
    ax.boxplot([fail_frac[n][0] for n in names], positions=pos, widths=0.9, showfliers=False,
               patch_artist=True, boxprops={"facecolor": "#9db4c8"}, medianprops={"color": "#222222"})
    ax.boxplot([fail_frac[n][1] for n in names], positions=pos + 1, widths=0.9, showfliers=False,
               patch_artist=True, boxprops={"facecolor": "#e8a598"}, medianprops={"color": "#222222"})
    ax.set_xticks(pos + 0.5, names)
    ax.set_ylabel("Fraction of dies that fail")
    ax.set_title("Failing-die fraction: real (blue) vs synthetic (red)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fail_fraction_real_vs_synthetic.png", dpi=150)
    plt.close(fig)

    print(f"overall detector AUC: {summary['overall_detector_auc']}")
    for name, r in results.items():
        print(f"{name}: {r}")


if __name__ == "__main__":
    main()
