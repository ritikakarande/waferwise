"""Turn LSWMD.pkl into a fast-loading .npz and create the fixed train/val/test split.

Usage:
    python scripts/prepare_data.py [--config configs/data.yaml]

The split file is created only once. If it already exists it is checked against
the processed data and left untouched, so the test set never changes.
"""

import argparse

import numpy as np

from waferwise.data import load_processed, load_splits, make_splits, preprocess, read_lswmd, save_processed, save_splits
from waferwise.utils import get_logger, load_config, repo_path, set_seed

log = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    processed_path = repo_path(cfg["processed_path"])
    if processed_path.exists():
        log.info("Using existing %s", processed_path)
        arrays = load_processed(processed_path)
    else:
        df = read_lswmd(cfg["raw_path"])
        log.info("Raw dataframe: %d rows, columns %s", len(df), list(df.columns))
        arrays = preprocess(df, cfg["image_size"])
        del df
        save_processed(arrays, processed_path)

    splits_path = repo_path(cfg["splits_path"])
    if splits_path.exists():
        splits = load_splits(splits_path, arrays["source_index"])
        log.info("Split file %s already exists and matches the data; not touching it", splits_path)
    else:
        # Group by map hash so exact duplicate wafers never end up in different splits.
        splits = make_splits(
            arrays["y"], cfg["seed"], cfg["val_fraction"], cfg["test_fraction"], groups=arrays["map_hash"]
        )
        save_splits(splits, arrays["source_index"], splits_path)
        log.info("Created %s", splits_path)

    for name, idx in splits.items():
        counts = np.bincount(arrays["y"][idx], minlength=len(arrays["classes"]))
        log.info("%-5s n=%6d  per class: %s", name, len(idx), dict(zip(arrays["classes"].tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
