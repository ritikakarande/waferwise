"""Calibrate the confidence gate on the validation split, using the deployed INT8 model.

Usage:
    python scripts/calibrate_gate.py [--config configs/deploy.yaml]

Writes gate.json (read by the C++ CLI) and gate_val.json. The threshold is also checked on held-out
halves of the validation split, so the reported accuracy is not only the one it was chosen on.
"""

import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from waferwise.deploy import predict_openvino
from waferwise.gate import gate_report, pick_threshold
from waferwise.train import load_data
from waferwise.utils import get_logger, load_config, repo_path, save_json

log = get_logger("calibrate_gate")


def held_out_check(probs: np.ndarray, y: np.ndarray, target: float, classes: list[str], repeats: int = 5) -> dict:
    """Pick the threshold on one half of the data and measure it on the other, in both directions."""
    results = []
    for seed in range(repeats):
        for half_a, half_b in StratifiedKFold(2, shuffle=True, random_state=seed).split(y, y):
            threshold = pick_threshold(probs[half_a], y[half_a], target)
            results.append(gate_report(probs[half_b], y[half_b], threshold, classes))
    return {
        "threshold_range": [min(r["threshold"] for r in results), max(r["threshold"] for r in results)],
        "auto_accuracy_mean": float(np.mean([r["auto_accuracy"] for r in results])),
        "auto_accuracy_min": float(min(r["auto_accuracy"] for r in results)),
        "review_share_mean": float(np.mean([r["review_share"] for r in results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/deploy.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = repo_path(cfg["out_dir"])

    data, splits, classes = load_data("configs/data.yaml")
    X, y = data["X"][splits["val"]], data["y"][splits["val"]]
    probs = predict_openvino(out_dir / "wafer_cnn_int8.xml", X)

    target = cfg["gate"]["target_auto_accuracy"]
    threshold = pick_threshold(probs, y, target)
    report = gate_report(probs, y, threshold, classes)
    report["held_out_check"] = held_out_check(probs, y, target, classes)
    gate = {"threshold": threshold, "target_auto_accuracy": target, "image_size": int(X.shape[1]), "classes": classes}
    save_json(gate, out_dir / "gate.json")
    save_json(report, out_dir / "gate_val.json")
    log.info("%s", {k: v for k, v in report.items() if k != "auto_metrics"})


if __name__ == "__main__":
    main()
