"""Tests for the browser game data in docs/assets (no dataset needed)."""

import json

import numpy as np
import onnx
import onnxruntime as ort

from waferwise.deploy import to_input
from waferwise.utils import repo_path
from waferwise.web import pack_codes, pick_game_wafers, unpack_codes

ASSETS = repo_path("docs/assets")


def test_packed_wafers_unpack_to_the_same_codes():
    X = np.random.default_rng(0).integers(0, 3, size=(5, 8, 8), dtype=np.uint8)

    packed = pack_codes(X)

    assert len(packed) == X.size // 4
    np.testing.assert_array_equal(unpack_codes(packed, 5, 8), X)


def test_game_draw_covers_every_class_without_repeats():
    y = np.repeat(np.arange(3), [50, 5, 100])

    picks = pick_game_wafers(y, per_class=10, seed=0)

    assert len(picks) == len(set(picks)) == 25
    assert np.array_equal(np.bincount(y[picks]), [10, 5, 10])


def test_site_assets_fit_together_and_the_model_runs():
    with open(ASSETS / "site.json", encoding="utf-8") as f:
        site = json.load(f)
    size, count = site["image_size"], len(site["wafer_labels"])
    wafers = unpack_codes((ASSETS / "wafers.bin").read_bytes(), count, size)

    assert 0.5 < site["threshold"] < 1
    assert set(site["wafer_labels"]) == set(range(len(site["classes"])))
    assert [p["threshold"] for p in site["curve"]] == sorted(p["threshold"] for p in site["curve"])

    session = ort.InferenceSession(str(ASSETS / "wafer_cnn.onnx"), providers=["CPUExecutionProvider"])
    probs = session.run(None, {"wafer": to_input(wafers[:16])})[0]
    assert probs.shape == (16, len(site["classes"]))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_published_model_has_no_export_debug_notes():
    model = onnx.load(ASSETS / "wafer_cnn.onnx")

    assert not any(node.metadata_props for node in model.graph.node)
    assert not model.graph.metadata_props and not model.metadata_props
    assert b"stack_trace" not in (ASSETS / "wafer_cnn.onnx").read_bytes()
