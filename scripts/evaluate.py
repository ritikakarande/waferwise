"""Evaluate a trained run on the validation or test split.

Usage:
    python scripts/evaluate.py --run outputs/baseline --split val
    python scripts/evaluate.py --run outputs/baseline --split test

The test split is for reporting final numbers only. Never use it to choose
models, thresholds or hyperparameters.
"""

import argparse

import numpy as np

from waferwise.evaluate import compute_metrics, format_metrics, predict, save_report
from waferwise.train import get_device, load_checkpoint, load_data, make_loader
from waferwise.utils import repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="folder containing model.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    device = get_device()
    run_dir = repo_path(args.run)
    model, cfg, classes = load_checkpoint(run_dir / "model.pt", device)
    data, splits, _ = load_data(cfg["data_config"])
    loader = make_loader(data, splits[args.split], 1024, False, cfg["train"], cfg["seed"])

    probs, y_true = predict(model, loader, device)
    metrics = compute_metrics(y_true, probs.argmax(1), classes)
    save_report(metrics, run_dir, args.split, classes, title_prefix=f"{cfg['experiment_name']}: ")
    np.save(run_dir / f"{args.split}_probs.npy", probs)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
