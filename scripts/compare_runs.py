"""Compare experiments across seeds: mean and spread of macro-F1 and per-class recall.

Usage:
    python scripts/compare_runs.py --split val --out outputs/compare_cleaning \
        --arm baseline outputs/baseline outputs/baseline_seed1 \
        --arm cleaned outputs/cleaned_seed42 outputs/cleaned_seed1
"""

import argparse
import json

import numpy as np
import pandas as pd

from waferwise.utils import repo_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", nargs="+", action="append", required=True,
                        metavar=("NAME", "RUN_DIR"), help="arm name followed by one run folder per seed")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", required=True, help="folder for comparison.csv / comparison.json")
    args = parser.parse_args()

    rows = []
    for name, *run_dirs in args.arm:
        for run in run_dirs:
            with open(repo_path(run) / f"metrics_{args.split}.json", encoding="utf-8") as f:
                m = json.load(f)
            row = {"arm": name, "run": run, "macro_f1": m["macro_f1"], "macro_recall": m["macro_recall"]}
            row.update({f"recall_{c}": v["recall"] for c, v in m["per_class"].items()})
            rows.append(row)
    runs = pd.DataFrame(rows)

    metrics = [c for c in runs.columns if c not in ("arm", "run")]
    summary = runs.groupby("arm", sort=False)[metrics].agg(["mean", "std"])
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / f"runs_{args.split}.csv", index=False)
    summary.to_csv(out_dir / f"comparison_{args.split}.csv")
    save_json(
        {arm: {m: {"mean": float(summary.loc[arm, (m, "mean")]),
                   "std": float(np.nan_to_num(summary.loc[arm, (m, "std")]))} for m in metrics}
         for arm in summary.index},
        out_dir / f"comparison_{args.split}.json",
    )

    table = pd.DataFrame({
        arm: [f"{summary.loc[arm, (m, 'mean')]:.3f} +/- {np.nan_to_num(summary.loc[arm, (m, 'std')]):.3f}"
              for m in metrics]
        for arm in summary.index
    }, index=metrics)
    print(f"{args.split} split, mean +/- std over seeds ({runs.groupby('arm').size().to_dict()} runs)")
    print(table.to_string())


if __name__ == "__main__":
    main()
