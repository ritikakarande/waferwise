"""Write wafer maps from one split as PNG files, the input format of the C++ CLI.

Usage:
    python scripts/export_wafer_images.py --split val --out outputs/deploy/wafers_val

Pixels are black (off wafer), grey (pass) or white (fail). File names keep the split order.
"""

import argparse

import cv2
import numpy as np

from waferwise.train import load_data
from waferwise.utils import repo_path

PIXEL_FOR_CODE = np.array([0, 128, 255], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="write only the first N wafers")
    args = parser.parse_args()

    data, splits, _ = load_data("configs/data.yaml")
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wafers = data["X"][splits[args.split]][: args.limit]
    for i, wafer in enumerate(wafers):
        if not cv2.imwrite(str(out_dir / f"wafer_{i:06d}.png"), PIXEL_FOR_CODE[wafer]):
            raise OSError(f"Could not write to {out_dir}")
    print(f"Wrote {len(wafers)} wafers to {out_dir}")


if __name__ == "__main__":
    main()
