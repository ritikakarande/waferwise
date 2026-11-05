"""Training loop with class-weighted loss and early stopping on validation macro-F1."""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from waferwise.data import WaferDataset, load_processed, load_splits
from waferwise.evaluate import compute_metrics, format_metrics, predict, save_report
from waferwise.models import build_model, count_parameters
from waferwise.utils import get_logger, load_config, repo_path, save_json, set_seed

log = get_logger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation and horizontal flip.

    Defect patterns (centre, ring, edge, scratch, ...) keep their class under
    these transforms, so this is a free and label-safe augmentation.
    """
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(-2, -1))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(-1,))
    return x


def class_weights(y: np.ndarray, num_classes: int, scheme: str) -> torch.Tensor:
    """Per-class loss weights, scaled so the average weight per training sample is 1."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if scheme == "none":
        w = np.ones(num_classes)
    elif scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weighting: {scheme}")
    w *= counts.sum() / (w * counts).sum()
    return torch.tensor(w, dtype=torch.float32)


def load_data(data_config: str) -> tuple[dict, dict, list[str]]:
    data_cfg = load_config(data_config)
    data = load_processed(data_cfg["processed_path"])
    splits = load_splits(data_cfg["splits_path"], data["source_index"])
    return data, splits, [str(c) for c in data["classes"]]


def make_loader(data: dict, indices: np.ndarray, batch_size: int, train: bool, cfg: dict, seed: int) -> DataLoader:
    dataset = WaferDataset(
        data["X"], data["y"], indices, transform=random_dihedral if (train and cfg.get("augment")) else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=cfg.get("num_workers", 0),
        generator=torch.Generator().manual_seed(seed) if train else None,
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, memory_format=torch.channels_last), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train", linewidth=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val", linewidth=2)
    axes[0].set_title("Weighted cross-entropy", loc="left")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], color="#c0392b", linewidth=2)
    axes[1].set_title("Validation macro-F1", loc="left")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def drop_listed_wafers(train_idx: np.ndarray, source_index: np.ndarray, drop_csv: str | Path) -> np.ndarray:
    """Remove training wafers marked ``drop == 1`` in a label-audit CSV (matched by LSWMD row number)."""
    audit = pd.read_csv(repo_path(drop_csv))
    drop_ids = audit.loc[audit["drop"] == 1, "source_index"].to_numpy()
    keep = ~np.isin(source_index[train_idx], drop_ids)
    log.info("Dropping %d training wafers listed in %s", (~keep).sum(), drop_csv)
    return train_idx[keep]


def add_rare_class_wafers(
    data: dict, train_idx: np.ndarray, tcfg: dict, classes: list[str], seed: int
) -> tuple[dict, np.ndarray]:
    """Extend the training set with extra rare-class wafers, as set in the config.

    ``train.oversample: {classes, extra_per_class}`` repeats randomly chosen real training wafers of
    those classes (with replacement; augmentation makes the repeats differ).
    ``train.synthetic: {path, classes, per_class}`` appends generated wafers of those classes.

    Returns the image/label arrays for the training loader and the indices into them. Only the
    training set is ever extended; validation and test use ``data`` unchanged.
    """
    rng = np.random.default_rng(seed)
    images, labels, indices = data["X"], data["y"], train_idx

    if over := tcfg.get("oversample"):
        extra = [
            rng.choice(train_idx[labels[train_idx] == classes.index(name)], over["extra_per_class"], replace=True)
            for name in over["classes"]
        ]
        indices = np.concatenate([indices, *extra])
        log.info("Oversampling: +%d repeated real wafers each for %s", over["extra_per_class"], over["classes"])

    if syn := tcfg.get("synthetic"):
        synthetic = np.load(repo_path(syn["path"]))
        picks = [
            rng.choice(np.flatnonzero(synthetic["y"] == classes.index(name)), syn["per_class"], replace=False)
            for name in syn["classes"]
        ]
        offset = len(images)  # synthetic wafers are stored after the real ones
        images = np.concatenate([images, synthetic["X"]])
        labels = np.concatenate([labels, synthetic["y"]])
        indices = np.concatenate([indices, offset + np.concatenate(picks)])
        log.info("Synthetic: +%d generated wafers each for %s (%s)", syn["per_class"], syn["classes"], syn["path"])

    return {"X": images, "y": labels}, indices


def train(cfg: dict, out_dir: str | Path, limit: int | None = None, train_indices: np.ndarray | None = None) -> dict:
    """Train a model from a config dict; saves checkpoint, history and val report into ``out_dir``.

    ``limit`` caps the number of training samples (for quick smoke tests). ``train_indices``
    replaces the official training split (used for cross-validation folds). If the config sets
    ``train.drop_list``, wafers flagged in that label-audit CSV are removed from training;
    ``train.oversample`` and ``train.synthetic`` add rare-class wafers (see ``add_rare_class_wafers``).
    Class weights are computed on the final training mix.
    """
    out_dir = repo_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    device = get_device()

    data, splits, classes = load_data(cfg["data_config"])
    train_idx = splits["train"] if train_indices is None else train_indices
    if tcfg.get("drop_list"):
        train_idx = drop_listed_wafers(train_idx, data["source_index"], tcfg["drop_list"])
    if limit is not None:
        train_idx = np.random.default_rng(cfg["seed"]).choice(train_idx, size=min(limit, len(train_idx)), replace=False)

    train_data, train_idx = add_rare_class_wafers(data, train_idx, tcfg, classes, cfg["seed"])

    train_loader = make_loader(train_data, train_idx, tcfg["batch_size"], True, tcfg, cfg["seed"])
    val_loader = make_loader(data, splits["val"], 1024, False, tcfg, cfg["seed"])

    # channels_last memory layout makes these convolutions ~2x faster on CPU.
    model = build_model(cfg["model"], len(classes)).to(device, memory_format=torch.channels_last)
    train_labels = train_data["y"][train_idx]
    log.info("Training mix: %s", dict(zip(classes, np.bincount(train_labels, minlength=len(classes)).tolist())))
    weights = class_weights(train_labels, len(classes), tcfg["class_weighting"])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    log.info(
        "Training %s on %s: %d params, %d train / %d val wafers",
        cfg["experiment_name"], device, count_parameters(model), len(train_idx), len(splits["val"]),
    )
    log.info("Class weights: %s", {c: round(float(w), 2) for c, w in zip(classes, weights)})

    history, best_f1, best_epoch = [], -1.0, 0
    checkpoint_path = out_dir / "model.pt"
    for epoch in range(1, tcfg["epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        probs, y_val = predict(model, val_loader, device)
        log_probs = torch.from_numpy(probs).clamp_min(1e-12).log()
        val_loss = nn.functional.nll_loss(log_probs, torch.from_numpy(y_val), weight=weights).item()
        val_f1 = compute_metrics(y_val, probs.argmax(1), classes)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": val_f1, "seconds": round(time.time() - start, 1)})
        log.info("epoch %2d  train_loss %.4f  val_loss %.4f  val_macro_f1 %.4f  (%.0fs)",
                 epoch, train_loss, val_loss, val_f1, history[-1]["seconds"])

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"model_state": model.state_dict(), "config": cfg, "classes": classes, "epoch": epoch},
                       checkpoint_path)
        elif epoch - best_epoch >= tcfg["early_stopping_patience"]:
            log.info("Early stopping: no improvement for %d epochs", tcfg["early_stopping_patience"])
            break

    save_json(history, out_dir / "history.json")
    plot_history(history, out_dir / "training_curves.png")

    # Final validation report from the best checkpoint.
    model, _, _ = load_checkpoint(checkpoint_path, device)
    probs, y_val = predict(model, val_loader, device)
    metrics = compute_metrics(y_val, probs.argmax(1), classes)
    metrics.update({"best_epoch": best_epoch, "epochs_run": len(history),
                    "train_seconds": round(sum(h["seconds"] for h in history), 1)})
    save_report(metrics, out_dir, "val", classes, title_prefix=f"{cfg['experiment_name']}: ")
    np.save(out_dir / "val_probs.npy", probs)
    log.info("Best epoch %d, validation results:\n%s", best_epoch, format_metrics(metrics))
    return metrics


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict, list[str]]:
    checkpoint = torch.load(repo_path(path), map_location=device, weights_only=True)
    cfg, classes = checkpoint["config"], checkpoint["classes"]
    model = build_model(cfg["model"], len(classes)).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg, classes
