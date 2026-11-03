"""Evaluation: macro-F1, per-class precision/recall, confusion matrix.

Plain accuracy is reported only alongside these: with 85% of wafers labelled
"none", a model that never finds a defect still scores 85% accuracy.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from waferwise.utils import repo_path, save_json


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return softmax probabilities (N, C) and true labels (N,) for every batch in ``loader``."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall.mean()),  # a.k.a. balanced accuracy
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_metrics(metrics: dict) -> str:
    """Human-readable table of the headline and per-class numbers."""
    lines = [
        f"macro-F1 {metrics['macro_f1']:.4f} | macro recall {metrics['macro_recall']:.4f} | "
        f"accuracy {metrics['accuracy']:.4f} | n={metrics['n']}",
        f"{'class':<10} {'recall':>7} {'precision':>9} {'f1':>7} {'support':>8}",
    ]
    for name, m in metrics["per_class"].items():
        lines.append(f"{name:<10} {m['recall']:>7.3f} {m['precision']:>9.3f} {m['f1']:>7.3f} {m['support']:>8d}")
    return "\n".join(lines)


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], path: str | Path, title: str) -> None:
    """Row-normalised confusion matrix (diagonal = per-class recall), with raw counts in each cell."""
    cm = np.asarray(cm)
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] == 0:
                continue
            color = "white" if row_share[i, j] > 0.6 else "#222222"
            ax.text(j, i, f"{row_share[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")
    fig.tight_layout()
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(metrics: dict, out_dir: str | Path, split: str, classes: list[str], title_prefix: str = "") -> None:
    """Write ``metrics_<split>.json`` and ``confusion_<split>.png`` into ``out_dir``."""
    out_dir = repo_path(out_dir)
    save_json(metrics, out_dir / f"metrics_{split}.json")
    title = f"{title_prefix}{split} confusion matrix (macro-F1 {metrics['macro_f1']:.3f})"
    plot_confusion_matrix(metrics["confusion_matrix"], classes, out_dir / f"confusion_{split}.png", title)
