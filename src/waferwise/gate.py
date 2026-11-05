"""Confidence gate: decide which wafers the model may classify alone and which go to a person.

A wafer is decided automatically when the model's top probability reaches a threshold; everything
else goes to review. The threshold is the lowest one that still reaches a target accuracy on the
automatically decided wafers of the calibration set. It is measured on the deployed (INT8) model,
so the softmax does not need to be well calibrated: only the ranking of confidences matters.
"""

import numpy as np

from waferwise.evaluate import compute_metrics


def pick_threshold(probs: np.ndarray, y: np.ndarray, target_accuracy: float) -> float:
    """Lowest confidence threshold whose automatically decided wafers reach ``target_accuracy``."""
    confidence = probs.max(axis=1)
    order = np.argsort(-confidence)
    correct = (probs.argmax(axis=1) == y)[order]
    accuracy = np.cumsum(correct) / np.arange(1, len(y) + 1)  # accuracy on the k most confident wafers
    reachable = np.flatnonzero(accuracy >= target_accuracy)
    if len(reachable) == 0:
        raise ValueError(f"No threshold reaches an accuracy of {target_accuracy}")
    return float(confidence[order[reachable[-1]]])


def gate_report(probs: np.ndarray, y: np.ndarray, threshold: float, classes: list[str], none_class: str = "none") -> dict:
    """How much work the gate takes on, and how good its automatic decisions are."""
    pred = probs.argmax(axis=1)
    auto = probs.max(axis=1) >= threshold
    wrong = pred != y
    none = classes.index(none_class)
    missed = (pred == none) & (y != none)  # a defect called clean
    return {
        "threshold": threshold,
        "n": int(len(y)),
        "review_share": float((~auto).mean()),
        "auto_accuracy": float((~wrong[auto]).mean()),
        "auto_metrics": compute_metrics(y[auto], pred[auto], classes),
        "errors_without_gate": int(wrong.sum()),
        "errors_after_gate": int(wrong[auto].sum()),
        "missed_defects_without_gate": int(missed.sum()),
        "missed_defects_after_gate": int(missed[auto].sum()),
        "errors_caught_by_review": float(wrong[~auto].sum() / max(wrong.sum(), 1)),
        "review_share_per_class": {name: float((~auto[y == c]).mean()) for c, name in enumerate(classes)},
    }
