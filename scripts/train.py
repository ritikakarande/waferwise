"""Train a wafer-map classifier from a YAML config. Results go to outputs/<experiment_name>/.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --epochs 1 --limit 5000   # quick smoke test
"""

import argparse

from waferwise.train import train
from waferwise.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--limit", type=int, help="use at most this many training wafers")
    parser.add_argument("--out", help="output folder (default: outputs/<experiment_name>)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, args.out or f"outputs/{cfg['experiment_name']}", limit=args.limit)


if __name__ == "__main__":
    main()
