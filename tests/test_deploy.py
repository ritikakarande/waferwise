"""Tests for export, quantisation and the confidence gate (no dataset needed)."""

import numpy as np
import openvino as ov
import torch

from waferwise.deploy import calibration_indices, export_onnx, predict_openvino, quantize_int8, to_input
from waferwise.gate import gate_report, pick_threshold
from waferwise.models import WaferCNN


def test_to_input_scales_codes_and_adds_a_channel():
    x = to_input(np.array([[[0, 1], [2, 1]]], dtype=np.uint8))
    assert x.shape == (1, 1, 2, 2) and x.dtype == np.float32
    assert np.array_equal(x[0, 0], [[0.0, 0.5], [1.0, 0.5]])


def test_calibration_sample_is_balanced_and_comes_from_training_only():
    y = np.array([0] * 500 + [1] * 30 + [2] * 200)
    train_idx = np.arange(0, 700)  # the last 30 rows are not training wafers
    idx = calibration_indices(y, train_idx, per_class=50, seed=0)
    assert np.array_equal(np.bincount(y[idx]), [50, 30, 50])
    assert idx.max() < 700 and len(set(idx)) == len(idx)


def test_openvino_export_matches_pytorch_and_int8_runs(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = WaferCNN(num_classes=9, channels=(4, 8, 8, 8)).eval()
    X = rng.integers(0, 3, size=(32, 64, 64), dtype=np.uint8)

    export_onnx(model, tmp_path / "m.onnx")
    fp32 = ov.convert_model(tmp_path / "m.onnx")
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(to_input(X))), dim=1).numpy()
    probs = predict_openvino(fp32, X, batch_size=7)  # a batch size the export did not see
    assert np.abs(probs - expected).max() < 1e-4
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    int8 = quantize_int8(fp32, X)
    assert "FakeQuantize" in {op.get_type_name() for op in int8.get_ops()}
    assert predict_openvino(int8, X).shape == (32, 9)


def test_threshold_is_the_lowest_that_reaches_the_target():
    probs = np.array([[0.99, 0.01], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    y = np.array([0, 0, 0, 1, 0])  # only the 0.7 wafer is wrong
    assert pick_threshold(probs, y, 1.0) == 0.8
    assert pick_threshold(probs, y, 0.8) == 0.6  # all five wafers: 4/5 correct


def test_gate_report_counts_errors_and_missed_defects():
    classes = ["Loc", "none"]
    probs = np.array([[0.95, 0.05], [0.05, 0.95], [0.40, 0.60], [0.10, 0.90]])
    y = np.array([0, 1, 0, 0])  # the last two are defects called "none"
    r = gate_report(probs, y, threshold=0.8, classes=classes)
    assert r["review_share"] == 0.25
    assert r["errors_without_gate"] == 2 and r["errors_after_gate"] == 1
    assert r["missed_defects_without_gate"] == 2 and r["missed_defects_after_gate"] == 1
    assert r["errors_caught_by_review"] == 0.5
    assert r["review_share_per_class"] == {"Loc": 1 / 3, "none": 0.0}
