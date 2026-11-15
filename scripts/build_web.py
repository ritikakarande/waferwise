"""Build the data for the browser game in docs/ (served by GitHub Pages).

Usage:
    python scripts/build_web.py

Writes to docs/assets/:
  wafer_cnn.onnx  the final model (FP32 ONNX), which the page runs with ONNX Runtime Web
  wafers.bin      sample wafers from the validation split, 4 pixels per byte (see waferwise.web)
  site.json       class names, the gate threshold, the gate's operating curve, the wafer labels and one
                  example wafer per class for the pattern guide

The browser runs FP32, not the INT8 OpenVINO model, so the gate threshold is calibrated again here with
the same rule, on the same validation split. The test split is not used.
"""

import json
import shutil

import numpy as np
import onnxruntime as ort

from waferwise.deploy import strip_onnx_metadata, to_input
from waferwise.gate import gate_report, pick_threshold
from waferwise.train import load_data
from waferwise.utils import get_logger, load_config, repo_path
from waferwise.web import gate_curve, pack_codes, pick_game_wafers

log = get_logger("build_web")
WAFERS_PER_CLASS = 30
SEED = 42


def onnx_probabilities(model_path, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return np.concatenate([
        session.run(None, {"wafer": to_input(X[i:i + batch_size])})[0] for i in range(0, len(X), batch_size)
    ])


def main() -> None:
    deploy_cfg = load_config("configs/deploy.yaml")
    out_dir = repo_path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "wafer_cnn.onnx"
    shutil.copy(repo_path(deploy_cfg["out_dir"]) / "wafer_cnn.onnx", model_path)
    strip_onnx_metadata(model_path)  # also for models exported before export_onnx stripped it

    data, splits, classes = load_data("configs/data.yaml")
    X, y = data["X"][splits["val"]], data["y"][splits["val"]]
    probs = onnx_probabilities(model_path, X)

    target = deploy_cfg["gate"]["target_auto_accuracy"]
    threshold = pick_threshold(probs, y, target)
    report = gate_report(probs, y, threshold, classes)
    curve = gate_curve(probs, y, np.append(np.arange(0.30, 0.996, 0.005), threshold), classes)
    curve.sort(key=lambda point: point["threshold"])

    picks = pick_game_wafers(y, WAFERS_PER_CLASS, SEED)
    (out_dir / "wafers.bin").write_bytes(pack_codes(X[picks]))
    # Pattern guide: for each class, the sample wafer the model is most sure belongs to its labelled class.
    sample_probs, sample_labels = probs[picks], y[picks]
    guide = [int(np.argmax(np.where(sample_labels == c, sample_probs[:, c], -1))) for c in range(len(classes))]

    site = {
        "image_size": int(X.shape[1]),
        "classes": classes,
        "threshold": round(float(threshold), 4),
        "target_auto_accuracy": target,
        "val_wafers": int(len(y)),
        "val_review_share": round(report["review_share"], 4),
        "val_auto_accuracy": round(report["auto_accuracy"], 5),
        "curve": curve,
        "wafer_labels": [int(label) for label in y[picks]],
        "guide_examples": guide,
    }
    with open(out_dir / "site.json", "w", encoding="utf-8") as f:
        json.dump(site, f, separators=(",", ":"))
    log.info("FP32 gate on val: threshold %.4f, review %.2f%%, auto accuracy %.4f; %d game wafers",
             threshold, 100 * report["review_share"], report["auto_accuracy"], len(picks))


if __name__ == "__main__":
    main()
