"""Out-of-fold predictions for every training wafer (input for the label audit).

The training split is cut into K folds (settings in the label-audit config). For each
fold a model is trained on the other K-1 folds and predicts the held-out fold, so every
training wafer gets a prediction from a model that never saw it. The val and test
splits are not touched, except that val is used for early stopping as usual.

Usage:
    python scripts/cross_val.py --config configs/audit_cnn_a.yaml               # all folds, one after another
    python scripts/cross_val.py --config configs/audit_cnn_a.yaml --fold 2 --threads 4   # one fold (run folds in parallel)
"""

import argparse

import numpy as np
import torch

from waferwise.evaluate import predict
from waferwise.label_audit import make_folds
from waferwise.train import get_device, load_checkpoint, load_data, make_loader, train
from waferwise.utils import load_config, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, help="run only this fold")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    audit_cfg = load_config(cfg["audit_config"])
    data, splits, _ = load_data(cfg["data_config"])
    train_pos = splits["train"]
    folds = make_folds(data["y"][train_pos], data["map_hash"][train_pos], audit_cfg["folds"], audit_cfg["seed"])

    out_root = repo_path(f"outputs/{cfg['experiment_name']}")
    device = get_device()
    for fold in [args.fold] if args.fold is not None else range(audit_cfg["folds"]):
        fold_dir = out_root / f"fold{fold}"
        train(cfg, fold_dir, train_indices=train_pos[folds != fold])

        held_out = train_pos[folds == fold]
        model, _, _ = load_checkpoint(fold_dir / "model.pt", device)
        loader = make_loader(data, held_out, 1024, False, cfg["train"], cfg["seed"])
        probs, _ = predict(model, loader, device)
        np.savez(fold_dir / "oof.npz", positions=held_out, probs=probs)
        print(f"fold {fold}: saved out-of-fold predictions for {len(held_out)} wafers")


if __name__ == "__main__":
    main()
