"""Find likely mislabelled training wafers and write the suspect list + review grids.

Needs the out-of-fold CNN predictions from scripts/cross_val.py. Outputs:
    audit/suspect_labels.csv          suspects (committed; rows with drop=1 are removed by configs/cleaned.yaml)
    outputs/label_audit/summary.json  counts, per-class breakdown, duplicate-conflict check
    outputs/label_audit/review_*.png  image grids for review by hand

Usage:
    python scripts/label_audit.py [--config configs/label_audit.yaml]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score

from waferwise.label_audit import audit_labels, cross_val_trees, make_folds, wafer_features
from waferwise.train import load_data
from waferwise.utils import get_logger, load_config, repo_path, save_json

log = get_logger("label_audit")
WAFER_CMAP = ListedColormap(["#f2f2f2", "#9db4c8", "#c0392b"])  # off-wafer, pass, fail


def load_oof(member_dir: str, train_pos: np.ndarray, k: int) -> np.ndarray:
    """Stitch the K out-of-fold prediction files back into one array aligned with ``train_pos``."""
    lookup = {p: i for i, p in enumerate(train_pos)}
    probs = None
    seen = np.zeros(len(train_pos), dtype=bool)
    for fold in range(k):
        npz = np.load(repo_path(member_dir) / f"fold{fold}" / "oof.npz")
        rows = np.array([lookup[p] for p in npz["positions"]])
        if probs is None:
            probs = np.zeros((len(train_pos), npz["probs"].shape[1]), dtype=np.float32)
        probs[rows] = npz["probs"]
        seen[rows] = True
    if not seen.all():
        raise ValueError(f"{member_dir}: {np.sum(~seen)} training wafers have no out-of-fold prediction")
    return probs


def plot_grid(X, rows: pd.DataFrame, classes, path, title, ncols=8):
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.45, nrows * 1.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(X[r["position"]], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"{classes[r['given']]}\n-> {classes[r['suggested']]}", fontsize=7.5, pad=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def duplicate_conflict_check(table: pd.DataFrame, hashes: np.ndarray) -> dict:
    """Pixel-identical wafers with different labels: at least one label in each group must be wrong."""
    df = table.assign(hash=hashes)
    n_labels = df.groupby("hash")["given"].transform("nunique")
    conflicts = df[n_labels > 1]
    groups = conflicts.groupby("hash")
    return {
        "conflicting_groups_in_train": int(groups.ngroups),
        "groups_with_a_cleanlab_flag": int(groups["cleanlab_issue"].any().sum()),
        "groups_with_a_dropped_wafer": int(groups["drop"].any().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/label_audit.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = repo_path(f"outputs/{cfg['experiment_name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data, splits, classes = load_data(cfg["data_config"])
    train_pos = splits["train"]
    y = data["y"][train_pos]
    folds = make_folds(y, data["map_hash"][train_pos], cfg["folds"], cfg["seed"])

    member_probs = {name: load_oof(path, train_pos, cfg["folds"]) for name, path in cfg["cnn_members"].items()}
    trees_path = out_dir / "oof_trees.npy"
    if trees_path.exists():
        member_probs["trees"] = np.load(trees_path)
    else:
        features = wafer_features(data["X"][train_pos])
        member_probs["trees"] = cross_val_trees(features, y, folds, cfg["seed"], **cfg["trees"])
        np.save(trees_path, member_probs["trees"])

    member_f1 = {n: round(float(f1_score(y, p.argmax(1), average="macro")), 4) for n, p in member_probs.items()}
    log.info("Out-of-fold macro-F1 per member: %s", member_f1)

    table = audit_labels(y, member_probs)
    table.insert(0, "position", train_pos)
    table.insert(1, "source_index", data["source_index"][train_pos])

    # Suspect list: everything either signal flagged, most suspicious first.
    suspects = table[table["cleanlab_issue"] | (table["models_disagreeing"] >= 2)].sort_values("label_quality")
    csv = suspects.drop(columns=["position"] + [c for c in suspects if c.startswith("pred_")]).copy()
    csv["given"] = [classes[i] for i in csv["given"]]
    csv["suggested"] = [classes[i] for i in csv["suggested"]]
    for col in ("cleanlab_issue", "all_models_agree_other", "drop"):
        csv[col] = csv[col].astype(int)
    csv = csv.round({"label_quality": 4, "given_prob": 4})
    csv_path = repo_path(cfg["suspects_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv.to_csv(csv_path, index=False)

    dropped = table[table["drop"]].sort_values("label_quality")
    flips = pd.crosstab(
        dropped["given"].map(lambda i: classes[i]), dropped["suggested"].map(lambda i: classes[i])
    )
    flips.to_csv(out_dir / "dropped_given_vs_suggested.csv")

    per_class = pd.DataFrame({
        "train_wafers": np.bincount(y, minlength=len(classes)),
        "cleanlab_flagged": np.bincount(y[table["cleanlab_issue"]], minlength=len(classes)),
        "dropped": np.bincount(y[table["drop"]], minlength=len(classes)),
    }, index=classes)
    per_class["dropped_%"] = (100 * per_class["dropped"] / per_class["train_wafers"]).round(2)
    per_class.to_csv(out_dir / "per_class.csv")

    summary = {
        "train_wafers": int(len(y)),
        "out_of_fold_macro_f1": member_f1,
        "cleanlab_flagged": int(table["cleanlab_issue"].sum()),
        "all_models_agree_on_other_class": int(table["all_models_agree_other"].sum()),
        "dropped_both_signals": int(table["drop"].sum()),
        "dropped_share_%": round(100 * float(table["drop"].mean()), 2),
        "per_class": per_class.to_dict(orient="index"),
        "duplicate_conflicts": duplicate_conflict_check(table, data["map_hash"][train_pos]),
    }
    save_json(summary, out_dir / "summary.json")

    plot_grid(data["X"], dropped.head(48), classes, out_dir / "review_top48.png",
              "Most suspicious training labels (given -> suggested by the models)")
    per_class_rows = pd.concat([dropped[dropped["given"] == c].head(8) for c in range(len(classes))])
    plot_grid(data["X"], per_class_rows, classes, out_dir / "review_by_class.png",
              "Top dropped suspects for each given label (up to 8 per class)")

    print(per_class.to_string())
    print("\nDropped wafers, given label (rows) vs suggested label (columns):")
    print(flips.to_string())
    print()
    for key in ("out_of_fold_macro_f1", "cleanlab_flagged", "all_models_agree_on_other_class",
                "dropped_both_signals", "dropped_share_%", "duplicate_conflicts"):
        print(f"{key}: {summary[key]}")
    log.info("Wrote %s and figures in %s", csv_path, out_dir)


if __name__ == "__main__":
    main()
