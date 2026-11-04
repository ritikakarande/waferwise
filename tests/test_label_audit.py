"""Tests for the label-audit helpers (synthetic data, no dataset needed)."""

import numpy as np
import pandas as pd

from waferwise.label_audit import audit_labels, make_folds, wafer_features
from waferwise.train import drop_listed_wafers


def test_folds_keep_duplicates_together_and_cover_every_class():
    y = np.repeat(np.arange(3), 100)
    groups = np.arange(300) // 2  # pairs of identical wafers
    folds = make_folds(y, groups, k=5, seed=0)
    assert set(folds) == set(range(5))
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    for f in range(5):
        assert len(set(y[folds == f])) == 3


def test_wafer_features_shape_and_range():
    X = np.random.default_rng(0).integers(0, 3, (10, 64, 64), dtype=np.uint8)
    feats = wafer_features(X, chunk=4)
    assert feats.shape == (10, 256 + 8 + 1)
    assert feats.min() >= 0 and feats.max() <= 1


def test_audit_finds_injected_label_errors():
    rng = np.random.default_rng(0)
    n, k = 3000, 3
    true = rng.integers(0, k, n)
    labels = true.copy()
    flipped = rng.choice(n, 60, replace=False)
    labels[flipped] = (true[flipped] + 1) % k

    def confident_model(noise):
        probs = np.full((n, k), 0.05) + rng.uniform(0, noise, (n, k))
        probs[np.arange(n), true] = 0.9
        return probs / probs.sum(1, keepdims=True)

    table = audit_labels(labels, {"a": confident_model(0.05), "b": confident_model(0.05), "c": confident_model(0.1)})
    dropped = set(np.flatnonzero(table["drop"]))
    caught = len(set(flipped) & dropped)
    assert caught >= 0.95 * len(flipped)  # nearly every injected error is caught
    assert len(dropped - set(flipped)) <= 5  # and almost nothing else
    assert (table.loc[flipped, "suggested"].to_numpy() == true[flipped]).all()


def test_drop_list_removes_only_flagged_wafers(tmp_path):
    source_index = np.array([10, 20, 30, 40, 50])
    csv = tmp_path / "suspects.csv"
    pd.DataFrame({"source_index": [20, 40], "drop": [1, 0]}).to_csv(csv, index=False)
    kept = drop_listed_wafers(np.arange(5), source_index, csv)
    assert list(source_index[kept]) == [10, 30, 40, 50]
