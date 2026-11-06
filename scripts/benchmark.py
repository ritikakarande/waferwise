"""Benchmark the deployed model: FP32 vs INT8, PyTorch vs OpenVINO, Python vs C++.

Usage:
    python scripts/benchmark.py [--config configs/benchmark.yaml]

Needs the exported model (scripts/export_model.py, scripts/calibrate_gate.py) and a built C++ CLI
(cpp/README.md). Writes benchmark.json to the output folder. This script measures speed only:
accuracy comes from deployed_test.json.
"""

import argparse
import itertools
import json
import multiprocessing
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openvino as ov
import psutil
import torch

from waferwise.benchmark import run_cpp_case, run_python_case, summarise_case, summarise_run
from waferwise.utils import get_logger, load_config, repo_path, save_json

log = get_logger("benchmark")


def build_cases(cfg: dict) -> list[dict]:
    cases = []
    for batch, threads in itertools.product(cfg["batch_sizes"], cfg["threads"]):
        cases.append({"runtime": "pytorch", "precision": "fp32", "batch": batch, "threads": threads})
        for runtime, precision in itertools.product(("openvino_python", "openvino_cpp"), cfg["precisions"]):
            cases.append({"runtime": runtime, "precision": precision, "batch": batch, "threads": threads})
    return cases


def case_name(case: dict) -> str:
    return f"{case['runtime']}_{case['precision']}_batch{case['batch']}_threads{case['threads']}"


def run_case(case: dict, cfg: dict, deploy_cfg: dict, wafer_dir: Path, output: Path) -> np.ndarray:
    deploy_dir = repo_path(deploy_cfg["out_dir"])
    gate_path = deploy_dir / "gate.json"
    model_path = deploy_dir / f"wafer_cnn_{case['precision']}.xml"
    if case["runtime"] == "openvino_cpp":
        return run_cpp_case(case, cfg, wafer_dir, gate_path, model_path, output)

    if case["runtime"] == "pytorch":
        model_path = repo_path(deploy_cfg["run"]) / "model.pt"
    with open(gate_path, encoding="utf-8") as f:
        gate = json.load(f)
    return run_python_case(case, sorted(wafer_dir.glob("*.png")), gate, model_path, output, cfg["run_seconds"])


def on_battery() -> bool:
    battery = psutil.sensors_battery()
    return bool(battery and not battery.power_plugged)


def power_plan() -> str:
    if platform.system() != "Windows":
        return "unknown"
    return subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout.strip()


def system_info() -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": ov.Core().get_property("CPU", "FULL_DEVICE_NAME"),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 1e9),
        "os": platform.platform(),
        "python": platform.python_version(),
        "openvino": ov.get_version(),
        "pytorch": torch.__version__,
        "pytorch_default_threads": torch.get_num_threads(),
        "power_plan": power_plan(),
        "on_battery_at_start": on_battery(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    deploy_cfg = load_config(cfg["deploy_config"])

    out_dir = repo_path(cfg["out_dir"])
    wafer_dir = out_dir / "wafers"
    if len(list(wafer_dir.glob("*.png"))) != cfg["wafers"]:
        subprocess.run(
            [sys.executable, str(repo_path("scripts/export_wafer_images.py")), "--split", "val",
             "--out", str(wafer_dir), "--limit", str(cfg["wafers"])],
            check=True,
        )

    info = system_info()
    log.info("%s", info)
    if info["on_battery_at_start"]:
        log.warning("Running on battery: the CPU is limited and these figures will not represent it")
    cases = build_cases(cfg)
    runs = {case_name(case): [] for case in cases}
    rng = np.random.default_rng(cfg["seed"])
    context = multiprocessing.get_context("spawn")
    for repeat in range(cfg["repeats"]):
        for i in rng.permutation(len(cases)):
            # A new process for every run, so no run inherits another library's idle thread pool.
            with context.Pool(1) as pool:
                timings = pool.apply(run_case, (cases[i], cfg, deploy_cfg, wafer_dir, out_dir / "last_run.csv"))
            runs[case_name(cases[i])].append(summarise_run(timings, cfg["warmup_seconds"]))
        log.info("Repeat %d of %d done", repeat + 1, cfg["repeats"])

    info["on_battery_at_end"] = on_battery()
    results = [{**case, **summarise_case(runs[case_name(case)])} for case in cases]
    save_json({"system": info, "config": cfg, "results": results}, out_dir / "benchmark.json")
    for row in results:
        log.info(
            "%-52s p50 %7.3f ms  p99 %7.3f ms  inference %7.0f/s  end to end %6.0f/s",
            case_name(row), row["latency_p50_ms"]["median"], row["latency_p99_ms"]["median"],
            row["inference_wafers_per_s"]["median"], row["end_to_end_wafers_per_s"]["median"],
        )


if __name__ == "__main__":
    main()
