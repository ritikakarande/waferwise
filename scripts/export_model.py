"""Export the final model to ONNX and OpenVINO, quantise it to INT8 and compare the two on validation.

Usage:
    python scripts/export_model.py [--config configs/deploy.yaml]

Writes wafer_cnn.onnx, wafer_cnn_fp32.xml/.bin, wafer_cnn_int8.xml/.bin and export_report.json to
the output folder. Calibration uses training wafers only. The test split is not touched here.
"""

import argparse

import numpy as np
import openvino as ov
import torch

from waferwise.deploy import calibration_indices, export_onnx, predict_openvino, quantize_int8, to_input
from waferwise.evaluate import compute_metrics, format_metrics
from waferwise.models import build_model
from waferwise.train import load_data
from waferwise.utils import get_logger, load_config, repo_path, save_json, set_seed

log = get_logger("export_model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/deploy.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = repo_path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(repo_path(cfg["run"]) / "model.pt", map_location="cpu", weights_only=True)
    train_cfg, classes = checkpoint["config"], checkpoint["classes"]
    model = build_model(train_cfg["model"], len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data, splits, _ = load_data(train_cfg["data_config"])
    X_val, y_val = data["X"][splits["val"]], data["y"][splits["val"]]

    onnx_path = out_dir / "wafer_cnn.onnx"
    export_onnx(model, onnx_path)
    fp32 = ov.convert_model(onnx_path)
    fp32_path = out_dir / "wafer_cnn_fp32.xml"
    ov.save_model(fp32, fp32_path, compress_to_fp16=False)

    calibration = calibration_indices(data["y"], splits["train"], cfg["quantization"]["samples_per_class"], cfg["seed"])
    int8 = quantize_int8(fp32, data["X"][calibration])
    int8_path = out_dir / "wafer_cnn_int8.xml"
    ov.save_model(int8, int8_path)

    with torch.no_grad():
        torch_probs = torch.softmax(model(torch.from_numpy(to_input(X_val))), dim=1).numpy()
    fp32_probs = predict_openvino(fp32_path, X_val)
    int8_probs = predict_openvino(int8_path, X_val)
    fp32_metrics = compute_metrics(y_val, fp32_probs.argmax(1), classes)
    int8_metrics = compute_metrics(y_val, int8_probs.argmax(1), classes)
    log.info("OpenVINO FP32 on val:\n%s", format_metrics(fp32_metrics))
    log.info("OpenVINO INT8 on val:\n%s", format_metrics(int8_metrics))

    report = {
        "calibration_wafers": len(calibration),
        "pytorch_vs_fp32_max_prob_difference": float(np.abs(torch_probs - fp32_probs).max()),
        "pytorch_vs_fp32_label_changes": int((torch_probs.argmax(1) != fp32_probs.argmax(1)).sum()),
        "fp32_vs_int8_max_prob_difference": float(np.abs(fp32_probs - int8_probs).max()),
        "fp32_vs_int8_label_changes": int((fp32_probs.argmax(1) != int8_probs.argmax(1)).sum()),
        "file_size_mb": {
            "fp32": (out_dir / "wafer_cnn_fp32.bin").stat().st_size / 1e6,
            "int8": (out_dir / "wafer_cnn_int8.bin").stat().st_size / 1e6,
        },
        "val": {"fp32": fp32_metrics, "int8": int8_metrics},
    }
    save_json(report, out_dir / "export_report.json")
    log.info("%s", {k: v for k, v in report.items() if k != "val"})


if __name__ == "__main__":
    main()
