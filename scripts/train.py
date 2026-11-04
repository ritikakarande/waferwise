"""Train a wafer-map classifier from a YAML config. Results go to outputs/<experiment_name>/.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --epochs 1 --limit 5000   # quick smoke test
    python scripts/train.py --config configs/cleaned.yaml --seed 1 --threads 4       # one of several parallel runs
"""

import argparse

import torch

from waferwise.train import train
from waferwise.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--limit", type=int, help="use at most this many training wafers")
    parser.add_argument("--seed", type=int, help="override the random seed")
    parser.add_argument("--threads", type=int, help="CPU threads for PyTorch (useful when running jobs in parallel)")
    parser.add_argument("--out", help="output folder (default: outputs/<experiment_name>[_seed<N>])")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.threads:
        torch.set_num_threads(args.threads)
    out = args.out or f"outputs/{cfg['experiment_name']}"
    if args.seed is not None:
        cfg["seed"] = args.seed
        out = args.out or f"{out}_seed{args.seed}"
    train(cfg, out, limit=args.limit)


if __name__ == "__main__":
    main()
