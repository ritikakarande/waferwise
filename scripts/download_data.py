"""Download the WM-811K dataset (LSWMD.pkl) from Kaggle into data/raw/.

Requires a Kaggle API token (https://www.kaggle.com/settings -> API -> Create New Token),
saved as ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json).

Usage:
    python scripts/download_data.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from waferwise.utils import load_config, repo_path


def find_kaggle_cli() -> str:
    # Prefer the CLI installed in the active virtual environment.
    scripts_dir = Path(sys.executable).parent
    for name in ("kaggle", "kaggle.exe"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which("kaggle")
    if found is None:
        sys.exit("Kaggle CLI not found. Install the requirements first: pip install -r requirements.txt")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pkl_path = repo_path(cfg["raw_path"])
    raw_dir = pkl_path.parent

    if pkl_path.exists() and not args.force:
        print(f"{pkl_path} already exists ({pkl_path.stat().st_size / 1e6:.0f} MB). Use --force to re-download.")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [find_kaggle_cli(), "datasets", "download", cfg["kaggle_dataset"], "-p", str(raw_dir), "--unzip"]
    print("Running:", " ".join(cmd[1:]))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            "Kaggle download failed. Check that your API token is at ~/.kaggle/kaggle.json "
            "and that you can open the dataset page on kaggle.com while logged in."
        )

    for leftover in raw_dir.glob("*.zip"):
        leftover.unlink()
    if not pkl_path.exists():
        sys.exit(f"Download finished but {pkl_path} was not found. Contents: {[p.name for p in raw_dir.iterdir()]}")
    print(f"OK: {pkl_path} ({pkl_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
