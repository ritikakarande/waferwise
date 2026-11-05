"""Evaluate the deployed FP32 and INT8 models and the confidence gate on one split.

Usage:
    python scripts/evaluate_deployed.py --split test

The test split is for reporting final numbers only: run it after the quantisation settings and
the gate threshold are fixed, and never change them because of what it shows.
"""

import argparse
import json

from waferwise.deploy import predict_openvino
from waferwise.evaluate import compute_metrics, format_metrics
from waferwise.gate import gate_report
from waferwise.train import load_data
from waferwise.utils import get_logger, load_config, repo_path, save_json

log = get_logger("evaluate_deployed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--config", default="configs/deploy.yaml")
    args = parser.parse_args()

    out_dir = repo_path(load_config(args.config)["out_dir"])
    with open(out_dir / "gate.json", encoding="utf-8") as f:
        gate = json.load(f)
    data, splits, classes = load_data("configs/data.yaml")
    X, y = data["X"][splits[args.split]], data["y"][splits[args.split]]

    fp32_probs = predict_openvino(out_dir / "wafer_cnn_fp32.xml", X)
    int8_probs = predict_openvino(out_dir / "wafer_cnn_int8.xml", X)
    report = {
        "fp32": compute_metrics(y, fp32_probs.argmax(axis=1), classes),
        "int8": compute_metrics(y, int8_probs.argmax(axis=1), classes),
        "gate_on_int8": gate_report(int8_probs, y, gate["threshold"], classes),
    }
    save_json(report, out_dir / f"deployed_{args.split}.json")
    log.info("FP32 on %s:\n%s", args.split, format_metrics(report["fp32"]))
    log.info("INT8 on %s:\n%s", args.split, format_metrics(report["int8"]))
    log.info("Gate on INT8: %s", {k: v for k, v in report["gate_on_int8"].items() if k != "auto_metrics"})


if __name__ == "__main__":
    main()
