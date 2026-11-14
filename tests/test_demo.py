"""Tests on the committed INT8 model and sample wafers (no dataset needed).

The C++ test runs only when WAFERWISE_INFER points to a built waferwise_infer binary:
    WAFERWISE_INFER=cpp/build/waferwise_infer pytest tests/test_demo.py
On Windows the OpenVINO and OpenCV DLL folders must be on PATH (see cpp/README.md).
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from waferwise.utils import repo_path

sys.path.insert(0, str(repo_path("scripts")))
from demo import classify  # noqa: E402

MODEL = repo_path("models/wafer_cnn_int8.xml")
GATE = repo_path("models/gate.json")
WAFERS = sorted(repo_path("demo/wafers").glob("*.png"))
CLI = os.environ.get("WAFERWISE_INFER")


@pytest.fixture(scope="module")
def gate() -> dict:
    with open(GATE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def python_rows(gate) -> list[dict]:
    return classify(MODEL, gate, WAFERS)


def test_demo_classifies_every_sample_wafer(gate, python_rows):
    assert len(python_rows) == len(WAFERS) == 12
    for row in python_rows:
        assert row["predicted"] in gate["classes"]
        assert row["decision"] == ("AUTO" if row["confidence"] >= gate["threshold"] else "REVIEW")

    # Two clear-cut wafers: the model must get them right and decide them alone on any CPU.
    by_file = {row["file"]: row for row in python_rows}
    for name, label in (("07_Near-full.png", "Near-full"), ("11_Donut.png", "Donut")):
        assert by_file[name]["predicted"] == label
        assert by_file[name]["decision"] == "AUTO"


@pytest.mark.skipif(not CLI, reason="set WAFERWISE_INFER to a built waferwise_infer binary")
def test_cpp_cli_matches_python(python_rows, tmp_path):
    output = tmp_path / "results.csv"
    subprocess.run(
        [str(repo_path(CLI)), "--model", str(MODEL), "--gate", str(GATE),
         "--input", str(repo_path("demo/wafers")), "--output", str(output)],
        check=True,
    )
    cpp = pd.read_csv(output)

    assert len(cpp) == len(python_rows)
    for cpp_row, python_row in zip(cpp.itertuples(), python_rows):
        assert cpp_row.file.replace("\\", "/").endswith(python_row["file"])
        assert cpp_row.predicted == python_row["predicted"]
        assert cpp_row.decision == python_row["decision"]
        assert cpp_row.confidence == pytest.approx(python_row["confidence"], abs=1e-5)
