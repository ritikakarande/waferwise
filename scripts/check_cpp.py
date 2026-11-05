"""Check that the C++ CLI gives the same answers as Python for the same model and gate.

Usage:
    python scripts/export_wafer_images.py --split val --out outputs/deploy/wafers_val
    waferwise_infer --model ... --gate ... --input outputs/deploy/wafers_val --output outputs/deploy/cpp_val.csv
    python scripts/check_cpp.py --csv outputs/deploy/cpp_val.csv --split val

Writes cpp_check_<split>.json next to the CSV.
"""

import argparse
import json

import numpy as np
import pandas as pd

from waferwise.deploy import predict_openvino
from waferwise.train import load_data
from waferwise.utils import load_config, repo_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, help="results file written by the C++ CLI")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--config", default="configs/deploy.yaml")
    args = parser.parse_args()

    out_dir = repo_path(load_config(args.config)["out_dir"])
    with open(out_dir / "gate.json", encoding="utf-8") as f:
        gate = json.load(f)
    cpp = pd.read_csv(repo_path(args.csv)).sort_values("file")

    data, splits, classes = load_data("configs/data.yaml")
    X = data["X"][splits[args.split]][: len(cpp)]
    probs = predict_openvino(out_dir / "wafer_cnn_int8.xml", X)
    predicted = np.array(classes)[probs.argmax(axis=1)]
    confidence = probs.max(axis=1)
    decision = np.where(confidence >= gate["threshold"], "AUTO", "REVIEW")

    report = {
        "wafers": len(cpp),
        "label_differences": int((cpp["predicted"].to_numpy() != predicted).sum()),
        "decision_differences": int((cpp["decision"].to_numpy() != decision).sum()),
        "max_confidence_difference": float(np.abs(cpp["confidence"].to_numpy() - confidence).max()),
    }
    save_json(report, repo_path(args.csv).parent / f"cpp_check_{args.split}.json")
    print(report)


if __name__ == "__main__":
    main()
