"""Sanity tests for preprocessing and the fixed splits.

Tests marked ``needs_data`` run only after `scripts/prepare_data.py` has produced the processed data.
"""

import numpy as np
import pytest

from waferwise.data import CLASSES, flatten_label, load_processed, load_splits, make_splits, resize_nearest
from waferwise.utils import load_config, repo_path

CFG = load_config("configs/data.yaml")
HAVE_DATA = repo_path(CFG["processed_path"]).exists() and repo_path(CFG["splits_path"]).exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run scripts/prepare_data.py first")


@pytest.fixture(scope="module")
def data():
    arrays = load_processed(CFG["processed_path"])
    arrays["splits"] = load_splits(CFG["splits_path"], arrays["source_index"])
    return arrays


def test_flatten_label():
    assert flatten_label(np.array([["Center"]], dtype=object)) == "Center"
    assert flatten_label(np.array([["none"]])) == "none"
    assert flatten_label(np.zeros((0, 0))) is None


def test_resize_nearest_keeps_categorical_values():
    rng = np.random.default_rng(0)
    for shape in [(25, 27), (6, 21), (212, 84), (64, 64)]:
        wafer = rng.integers(0, 3, size=shape, dtype=np.uint8)
        out = resize_nearest(wafer, 64)
        assert out.shape == (64, 64)
        assert set(np.unique(out)) <= {0, 1, 2}


def test_make_splits_is_deterministic_and_disjoint():
    y = np.repeat(np.arange(len(CLASSES)), 40)
    a = make_splits(y, seed=42, val_fraction=0.15, test_fraction=0.15)
    b = make_splits(y, seed=42, val_fraction=0.15, test_fraction=0.15)
    for name in a:
        assert np.array_equal(a[name], b[name])
    assert sum(len(v) for v in a.values()) == len(y)
    assert not set(a["train"]) & set(a["val"]) and not set(a["train"]) & set(a["test"])


@needs_data
def test_splits_do_not_overlap_and_cover_everything(data):
    s = data["splits"]
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not train & val and not train & test and not val & test
    assert len(train) + len(val) + len(test) == len(data["y"])


def test_make_splits_keeps_groups_together():
    y = np.repeat(np.arange(len(CLASSES)), 40)
    groups = np.arange(len(y)) // 2  # pairs of identical wafers
    s = make_splits(y, seed=0, val_fraction=0.15, test_fraction=0.15, groups=groups)
    for a in s:
        for b in s:
            if a != b:
                assert not set(groups[s[a]]) & set(groups[s[b]])


@needs_data
def test_no_duplicate_wafer_in_two_splits(data):
    hashes = {name: set(data["map_hash"][idx].tolist()) for name, idx in data["splits"].items()}
    assert not hashes["train"] & hashes["test"]
    assert not hashes["train"] & hashes["val"]
    assert not hashes["val"] & hashes["test"]


@needs_data
def test_every_class_in_every_split(data):
    for name, idx in data["splits"].items():
        counts = np.bincount(data["y"][idx], minlength=len(CLASSES))
        assert (counts > 0).all(), f"{name} is missing a class: {counts}"


@needs_data
def test_split_proportions(data):
    n = len(data["y"])
    assert abs(len(data["splits"]["test"]) / n - CFG["test_fraction"]) < 0.01
    assert abs(len(data["splits"]["val"]) / n - CFG["val_fraction"]) < 0.01


@needs_data
def test_processed_arrays_are_valid(data):
    X, y = data["X"], data["y"]
    size = CFG["image_size"]
    assert X.shape == (len(y), size, size) and X.dtype == np.uint8
    assert set(np.unique(X)) <= {0, 1, 2}
    assert y.min() >= 0 and y.max() < len(CLASSES)
    assert list(data["classes"]) == CLASSES
