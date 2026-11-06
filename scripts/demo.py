"""Classify the sample wafers in demo/wafers with the INT8 model and the confidence gate.

Usage:
    python scripts/demo.py

Needs no dataset: the model and the sample wafers are in the repository. File names end with the
wafer's dataset label, so the output can show when the model is wrong.
"""

import json
from pathlib import Path

import numpy as np
import openvino as ov

from waferwise.deploy import load_wafer_png
from waferwise.utils import repo_path


def classify(model_path: Path, gate: dict, paths: list[Path]) -> list[dict]:
    wafers = np.stack([load_wafer_png(path, gate["image_size"]) for path in paths])[:, None]
    model = ov.Core().compile_model(model_path, "CPU")
    probabilities = model(wafers)[model.output(0)]
    rows = []
    for path, row in zip(paths, probabilities):
        best = int(row.argmax())
        rows.append({
            "file": path.name,
            "label": path.stem.split("_", 1)[1],
            "predicted": gate["classes"][best],
            "confidence": float(row[best]),
            "decision": "AUTO" if row[best] >= gate["threshold"] else "REVIEW",
        })
    return rows


def main() -> None:
    model_dir = repo_path("models")
    with open(model_dir / "gate.json", encoding="utf-8") as f:
        gate = json.load(f)
    paths = sorted(repo_path("demo/wafers").glob("*.png"))
    rows = classify(model_dir / "wafer_cnn_int8.xml", gate, paths)

    print(f"{'wafer':<22}{'label':<12}{'predicted':<12}{'confidence':>10}  decision")
    for row in rows:
        mark = "" if row["label"] == row["predicted"] else "  (wrong)"
        print(f"{row['file']:<22}{row['label']:<12}{row['predicted']:<12}{row['confidence']:>10.3f}  {row['decision']}{mark}")

    automatic = [row for row in rows if row["decision"] == "AUTO"]
    correct = sum(row["label"] == row["predicted"] for row in automatic)
    print(f"\n{len(automatic)} decided automatically ({correct} correct), {len(rows) - len(automatic)} sent to review "
          f"(threshold {gate['threshold']:.3f})")


if __name__ == "__main__":
    main()
