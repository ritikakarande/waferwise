"""Sanity tests for the model, the metrics and the training helpers (no data needed)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from waferwise.data import CLASSES, WaferDataset
from waferwise.evaluate import compute_metrics, predict
from waferwise.models import WaferCNN, count_parameters
from waferwise.train import class_weights, random_dihedral


def test_majority_class_predictor_gets_high_accuracy_but_low_macro_f1():
    y_true = np.array([8] * 90 + list(range(8)) + [0, 1])  # 90 of 100 are "none"
    y_pred = np.full_like(y_true, 8)
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == 0.9
    assert m["macro_f1"] < 0.15
    assert m["per_class"]["none"]["recall"] == 1.0
    assert m["per_class"]["Near-full"]["recall"] == 0.0


def test_perfect_predictions():
    y = np.repeat(np.arange(len(CLASSES)), 3)
    m = compute_metrics(y, y, CLASSES)
    assert m["macro_f1"] == 1.0 and m["macro_recall"] == 1.0
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (len(CLASSES), len(CLASSES))
    assert np.array_equal(cm, np.diag(np.full(len(CLASSES), 3)))


def test_class_weights_average_one_per_sample_and_favour_rare_classes():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    for scheme in ("none", "inverse", "inverse_sqrt"):
        w = class_weights(y, 3, scheme).numpy()
        counts = np.bincount(y)
        assert np.isclose((w * counts).sum() / counts.sum(), 1.0)
    w = class_weights(y, 3, "inverse_sqrt").numpy()
    assert w[2] > w[1] > w[0]


def test_model_output_shape_and_size():
    model = WaferCNN(num_classes=len(CLASSES))
    out = model(torch.zeros(4, 1, 64, 64))
    assert out.shape == (4, len(CLASSES))
    assert count_parameters(model) < 300_000


def test_random_dihedral_keeps_shape_and_values():
    x = torch.randint(0, 3, (1, 64, 64)).float() / 2
    for _ in range(10):
        y = random_dihedral(x)
        assert y.shape == x.shape
        assert torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values)


def test_predict_returns_probabilities():
    images = np.random.default_rng(0).integers(0, 3, (10, 64, 64), dtype=np.uint8)
    labels = np.arange(10) % len(CLASSES)
    loader = DataLoader(WaferDataset(images, labels), batch_size=4)
    probs, y = predict(WaferCNN(num_classes=len(CLASSES)), loader)
    assert probs.shape == (10, len(CLASSES))
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)
    assert np.array_equal(y, labels)


def test_rare_class_wafers_are_added_to_training_only(tmp_path):
    from waferwise.train import add_rare_class_wafers

    classes = ["a", "b", "c"]
    data = {"X": np.zeros((30, 4, 4), dtype=np.uint8), "y": np.array([0] * 20 + [1] * 5 + [2] * 5)}
    train_idx = np.arange(25)  # rows 25-29 are "val" and must never be used
    synthetic = tmp_path / "synthetic.npz"
    np.savez(synthetic, X=np.full((12, 4, 4), 2, dtype=np.uint8), y=np.array([1] * 6 + [2] * 6))
    tcfg = {
        "oversample": {"classes": ["b"], "extra_per_class": 10},
        "synthetic": {"path": str(synthetic), "classes": ["c"], "per_class": 4},
    }
    out, idx = add_rare_class_wafers(data, train_idx, tcfg, classes, seed=0)

    assert len(idx) == 25 + 10 + 4
    repeats, synth = idx[25:35], idx[35:]
    assert set(repeats) <= set(range(20, 25))      # repeats are real class-b training wafers
    assert (synth >= 30).all() and (out["y"][synth] == 2).all() and (out["X"][synth] == 2).all()
    assert not np.isin(idx, np.arange(25, 30)).any()  # no validation rows leak in
    assert np.array_equal(out["X"][:30], data["X"]) and np.array_equal(out["y"][:30], data["y"])
