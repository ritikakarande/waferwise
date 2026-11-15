"""Data for the browser game in docs/: sample wafers in a compact format, and the gate's operating curve.

Wafer maps hold three codes (0 off wafer, 1 pass, 2 fail), so four pixels fit in one byte. docs/js/wafers.js
unpacks them the same way.
"""

import numpy as np

from waferwise.gate import gate_report


def pack_codes(X: np.ndarray) -> bytes:
    """(N, S, S) codes 0..2 -> bytes, 4 pixels per byte, first pixel in the lowest two bits."""
    flat = X.astype(np.uint8).reshape(-1)
    if flat.max(initial=0) > 2 or len(flat) % 4:
        raise ValueError("expected codes 0..2 and a pixel count divisible by 4")
    quads = flat.reshape(-1, 4)
    return (quads[:, 0] | quads[:, 1] << 2 | quads[:, 2] << 4 | quads[:, 3] << 6).astype(np.uint8).tobytes()


def unpack_codes(data: bytes, count: int, size: int) -> np.ndarray:
    """Inverse of pack_codes."""
    packed = np.frombuffer(data, dtype=np.uint8)
    quads = np.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1)
    return quads.reshape(count, size, size)


def pick_game_wafers(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """A seeded random draw of up to per_class wafers of every class, in a random order (not picked by hand)."""
    rng = np.random.default_rng(seed)
    picks = [rng.choice(np.flatnonzero(y == c), size=min(per_class, int(np.sum(y == c))), replace=False)
             for c in np.unique(y)]
    return rng.permutation(np.concatenate(picks))


def gate_curve(probs: np.ndarray, y: np.ndarray, thresholds: np.ndarray, classes: list[str]) -> list[dict]:
    """What the gate does at each threshold: share sent to review, accuracy when automatic, errors left."""
    curve = []
    for threshold in thresholds:
        report = gate_report(probs, y, float(threshold), classes)
        curve.append({
            "threshold": round(float(threshold), 4),
            "review_share": round(report["review_share"], 4),
            "auto_accuracy": round(report["auto_accuracy"], 5),
            "wrong_automatic": report["errors_after_gate"],
            "missed_defects": report["missed_defects_after_gate"],
        })
    return curve
