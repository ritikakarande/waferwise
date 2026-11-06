"""Measure the deployed model: the same job in PyTorch, OpenVINO from Python and the C++ CLI.

The job matches the C++ CLI step by step: read PNG wafers, run the model, apply the gate, write
one CSV row per wafer. Each run returns one row of timings per batch, in the same format as the
CLI's --timings file, so all runtimes go through the same statistics.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

from waferwise.deploy import load_wafer_png
from waferwise.models import build_model
from waferwise.utils import repo_path


def openvino_predictor(model_path: Path, threads: int):
    config = {"INFERENCE_NUM_THREADS": threads} if threads > 0 else {}
    request = ov.Core().compile_model(model_path, "CPU", config).create_infer_request()

    def predict(x: np.ndarray) -> np.ndarray:
        request.set_input_tensor(ov.Tensor(x, shared_memory=True))
        request.infer()
        return request.get_output_tensor().data

    return predict


def pytorch_predictor(checkpoint_path: Path, threads: int):
    if threads > 0:
        torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"]["model"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(memory_format=torch.channels_last).eval()

    @torch.inference_mode()
    def predict(x: np.ndarray) -> np.ndarray:
        wafers = torch.from_numpy(x).contiguous(memory_format=torch.channels_last)
        return torch.softmax(model(wafers), dim=1).numpy()

    return predict


def run_python_case(
    case: dict, paths: list[Path], gate: dict, model_path: Path, output: Path, seconds: float
) -> np.ndarray:
    """Run the job in Python for a fixed time, cycling through the wafers. Returns timings (batches, 4)."""
    build = pytorch_predictor if case["runtime"] == "pytorch" else openvino_predictor
    predict = build(model_path, case["threads"])
    size, classes, threshold = gate["image_size"], gate["classes"], gate["threshold"]
    buffer = np.empty((case["batch"], 1, size, size), dtype=np.float32)
    timings = []
    run_start = time.perf_counter()
    first = 0
    with open(output, "w", encoding="utf-8") as csv:
        csv.write("file,predicted,confidence,decision\n")
        while time.perf_counter() - run_start < seconds:
            batch_paths = [paths[(first + i) % len(paths)] for i in range(case["batch"])]
            first += case["batch"]
            load_start = time.perf_counter()
            for i, path in enumerate(batch_paths):
                buffer[i, 0] = load_wafer_png(path, size)
            infer_start = time.perf_counter()
            probabilities = predict(buffer[:len(batch_paths)])
            infer_end = time.perf_counter()
            for path, row in zip(batch_paths, probabilities):
                best = int(row.argmax())
                decision = "AUTO" if row[best] >= threshold else "REVIEW"
                csv.write(f"{path},{classes[best]},{row[best]:.6f},{decision}\n")
            timings.append((
                len(batch_paths),
                1000 * (infer_start - load_start),
                1000 * (infer_end - infer_start),
                1000 * (time.perf_counter() - load_start),
            ))
    return np.array(timings)


def run_cpp_case(case: dict, cfg: dict, wafer_dir: Path, gate_path: Path, model_path: Path, output: Path) -> np.ndarray:
    """Run the job with the C++ CLI for a fixed time. Returns timings (batches, 4)."""
    timings_path = output.with_suffix(".timings.csv")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(repo_path(d)) for d in cfg["cpp_dll_dirs"]] + [env["PATH"]])
    subprocess.run(
        [
            str(repo_path(cfg["cpp_binary"])), "--model", str(model_path), "--gate", str(gate_path),
            "--input", str(wafer_dir), "--output", str(output), "--batch", str(case["batch"]),
            "--threads", str(case["threads"]), "--timings", str(timings_path), "--seconds", str(cfg["run_seconds"]),
        ],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )
    return np.loadtxt(timings_path, delimiter=",", skiprows=1, ndmin=2)


def summarise_run(timings: np.ndarray, warmup_seconds: float) -> dict[str, float]:
    """Speed figures for one run, leaving out the warm-up time and a short final batch.

    Warm-up is a time, not a share of the wafers: laptop CPUs boost for the first seconds of a load,
    so a fast run could otherwise finish inside the boost and look better than it can sustain.
    """
    wafers, infer_ms, total_ms = timings[:, 0], timings[:, 2], timings[:, 3]
    keep = (wafers == wafers[0]) & (np.cumsum(total_ms) > 1000 * warmup_seconds)
    if not keep.any():
        raise ValueError(f"Run is shorter than the {warmup_seconds} s warm-up")
    return {
        "latency_p50_ms": float(np.percentile(infer_ms[keep], 50)),
        "latency_p99_ms": float(np.percentile(infer_ms[keep], 99)),
        "inference_wafers_per_s": float(1000 * wafers[keep].sum() / infer_ms[keep].sum()),
        "end_to_end_wafers_per_s": float(1000 * wafers[keep].sum() / total_ms[keep].sum()),
    }


def summarise_case(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Median of each figure over the repeats of one case, with the lowest and highest run."""
    return {
        key: {
            "median": float(np.median([r[key] for r in runs])),
            "min": float(min(r[key] for r in runs)),
            "max": float(max(r[key] for r in runs)),
        }
        for key in runs[0]
    }
