"""Create a second, lot-grouped split for a robustness check of the final model.

Usage:
    python scripts/make_lot_split.py

The main split (splits/splits_seed42.npz) keeps exact duplicate maps together, but wafers from one
lot can land in both train and test. Wafers in a lot share process history and often a defect
pattern, so test scores may be optimistic for a lot the model has never seen. This split keeps
every lot, and every copy of a duplicated map, inside a single split.

It is used only for the check in README ("Does the model work on new lots?"). The main split and
its test set stay the reported ones.
"""

import numpy as np

from waferwise.data import connected_groups, load_processed, make_group_splits, save_splits
from waferwise.utils import get_logger, load_config, repo_path, save_json

log = get_logger("make_lot_split")
OUT = "splits/splits_lot_seed42.npz"


def main() -> None:
    cfg = load_config("configs/data.yaml")
    data = load_processed(cfg["processed_path"])
    y, lots, classes = data["y"], data["lot_name"], [str(c) for c in data["classes"]]

    groups = connected_groups(lots, data["map_hash"])
    splits = make_group_splits(y, groups, cfg["seed"], cfg["val_fraction"], cfg["test_fraction"])
    save_splits(splits, data["source_index"], OUT)

    lots_per_split = [set(lots[idx]) for idx in splits.values()]
    assert not (lots_per_split[0] & lots_per_split[1] or lots_per_split[0] & lots_per_split[2]
                or lots_per_split[1] & lots_per_split[2]), "a lot is in two splits"

    summary = {
        "groups": int(groups.max() + 1),
        "lots": len(set(lots)),
        "share": {name: round(len(idx) / len(y), 4) for name, idx in splits.items()},
        "class_counts": {
            name: {c: int(np.sum(y[idx] == i)) for i, c in enumerate(classes)} for name, idx in splits.items()
        },
    }
    out_dir = repo_path("outputs/lot_split")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / "split_summary.json")
    log.info("Wrote %s: %s", OUT, summary)


if __name__ == "__main__":
    main()
