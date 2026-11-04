"""Label audit: find training wafers whose given label is probably wrong.

Two independent signals are combined:

1. Confident learning (cleanlab). Every training wafer gets an out-of-sample
   prediction from K-fold cross-validation, i.e. from a model that never saw
   that wafer. cleanlab estimates how often each class is confused with each
   other class and flags wafers whose given label is unlikely under that model.
2. Ensemble disagreement. Several different models (two CNNs and a gradient-
   boosted tree model on hand-made features) vote independently. A wafer is a
   strong suspect when every model agrees on the same, different class.

A wafer is dropped from training only when both signals agree.
"""

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from waferwise.utils import get_logger

log = get_logger(__name__)


def make_folds(y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id (0..k-1) per sample, stratified by label; identical wafers (same group) share a fold."""
    folds = np.empty(len(y), dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        folds[held_out] = fold
    return folds


def wafer_features(X: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Simple hand-made features for the tree model.

    - failing-die density on a 16x16 grid (256 values)
    - failing-die share in 8 concentric rings, centre to edge (8 values)
    - overall failing-die share (1 value)
    """
    size = X.shape[1]
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2) / (size / 2)
    ring = np.minimum((radius * 8).astype(int), 7)
    ring_masks = [(ring == r) for r in range(8)]

    out = []
    for start in range(0, len(X), chunk):
        x = X[start:start + chunk]
        fail = (x == 2).astype(np.float32)
        on = (x > 0).astype(np.float32)
        n = len(x)
        grid = fail.reshape(n, 16, size // 16, 16, size // 16).mean(axis=(2, 4)).reshape(n, -1)
        rings = np.stack(
            [fail[:, m].sum(1) / np.maximum(on[:, m].sum(1), 1) for m in ring_masks], axis=1
        )
        total = fail.sum((1, 2)) / np.maximum(on.sum((1, 2)), 1)
        out.append(np.concatenate([grid, rings, total[:, None]], axis=1))
    return np.concatenate(out)


def cross_val_trees(features: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int, **params) -> np.ndarray:
    """Out-of-fold class probabilities from a gradient-boosted tree model."""
    num_classes = int(y.max()) + 1
    probs = np.zeros((len(y), num_classes), dtype=np.float32)
    for fold in np.unique(folds):
        held_out = folds == fold
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **params)
        model.fit(features[~held_out], y[~held_out])
        probs[held_out] = model.predict_proba(features[held_out])
        log.info("Tree model fold %d done", fold)
    return probs


def audit_labels(labels: np.ndarray, member_probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Combine cleanlab and ensemble disagreement into one table (one row per wafer).

    Columns: suggested label (ensemble vote), label quality score (low = suspicious),
    cleanlab flag, number of models disagreeing with the given label, whether all
    models agree on the same other class, and the final ``drop`` decision.
    """
    names = list(member_probs)
    ensemble = np.mean([member_probs[n] for n in names], axis=0)
    ensemble = ensemble / ensemble.sum(1, keepdims=True)

    member_preds = np.stack([member_probs[n].argmax(1) for n in names], axis=1)
    disagree = (member_preds != labels[:, None]).sum(1)
    unanimous_other = (member_preds == member_preds[:, :1]).all(1) & (member_preds[:, 0] != labels)

    cleanlab_issue = find_label_issues(labels, ensemble, filter_by="prune_by_noise_rate")
    table = pd.DataFrame({
        "given": labels,
        "suggested": ensemble.argmax(1),
        "label_quality": get_label_quality_scores(labels, ensemble),
        "given_prob": ensemble[np.arange(len(labels)), labels],
        "cleanlab_issue": cleanlab_issue,
        "models_disagreeing": disagree,
        "all_models_agree_other": unanimous_other,
    })
    for n in names:
        table[f"pred_{n}"] = member_probs[n].argmax(1)
    table["drop"] = table["cleanlab_issue"] & table["all_models_agree_other"]
    return table
