"""Export a trained model to ONNX and OpenVINO, quantise it to INT8 and run it from Python.

The exported models take a float tensor (N, 1, 64, 64) in [0, 1] and return class
probabilities (N, C), so the softmax is part of the model and C++ does not need it.
"""

from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
from torch import nn


# PNG wafer maps, the input format of the C++ CLI: black off wafer, grey pass, white fail.
PIXEL_FOR_CODE = np.array([0, 128, 255], dtype=np.uint8)
VALUE_FOR_PIXEL = np.where(np.arange(256) < 64, 0.0, np.where(np.arange(256) < 192, 0.5, 1.0)).astype(np.float32)


def export_onnx(model: nn.Module, path: Path, image_size: int = 64) -> None:
    """Write the model + softmax as ONNX with a variable batch size."""
    wrapped = nn.Sequential(model, nn.Softmax(dim=1)).eval()
    batch = torch.export.Dim("batch", min=1, max=4096)
    torch.onnx.export(
        wrapped,
        (torch.zeros(2, 1, image_size, image_size),),
        path,
        input_names=["wafer"],
        output_names=["probabilities"],
        dynamic_shapes=({0: batch},),
        external_data=False,
        verbose=False,  # its progress messages use emoji, which crash on a Windows console
    )


def to_input(X: np.ndarray) -> np.ndarray:
    """Wafer maps (N, S, S) with codes 0/1/2 -> float model input (N, 1, S, S) in [0, 1]."""
    return (X.astype(np.float32) / 2.0)[:, None]


def load_wafer_png(path: Path | str, size: int = 64) -> np.ndarray:
    """Read one PNG wafer map as a float image (S, S), the same steps as load_wafer in cpp/src/main.cpp."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Cannot read {path}")
    return VALUE_FOR_PIXEL[cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST_EXACT)]


def predict_openvino(model: ov.Model | str | Path, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Class probabilities (N, C) from an OpenVINO model or a path to one."""
    core = ov.Core()
    compiled = core.compile_model(model, "CPU")
    output = compiled.output(0)
    return np.concatenate([
        compiled(to_input(X[i:i + batch_size]))[output] for i in range(0, len(X), batch_size)
    ])


def calibration_indices(y: np.ndarray, train_idx: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """A class-balanced sample of training wafers (a random sample would be 85% "none")."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in np.unique(y[train_idx]):
        members = train_idx[y[train_idx] == c]
        picks.append(rng.choice(members, size=min(per_class, len(members)), replace=False))
    return np.concatenate(picks)


def quantize_int8(model: ov.Model, X_calibration: np.ndarray) -> ov.Model:
    """Post-training INT8 quantisation: NNCF measures activation ranges on the calibration wafers."""
    batches = [to_input(X_calibration[i:i + 1]) for i in range(len(X_calibration))]
    # Full bias correction (not the fast default): on validation it cut changed labels from 139 to 47.
    return nncf.quantize(model, nncf.Dataset(batches), subset_size=len(batches), fast_bias_correction=False)
