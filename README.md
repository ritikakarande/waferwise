# waferwise

Wafer-map defect classification on WM-811K, from noisy, imbalanced data to a model that runs in real time on an Intel CPU.

> The data was dirty and imbalanced. I fixed the labels, created the rare defects that were missing, then shipped a model that runs in real time on an Intel CPU and flags what it's unsure about.

**Status:** work in progress. Results, benchmarks and a demo will be added here.

## Pipeline

1. Baseline defect-pattern classifier (PyTorch)
2. Label audit with confident learning (cleanlab) to find likely mislabelled wafers
3. Synthetic rare defects from a small conditional generative model
4. Deployment: ONNX -> OpenVINO, INT8 with NNCF, C++ inference CLI, confidence gate
5. Benchmarks: FP32 vs INT8, Python vs C++

## Setup

Python 3.10+ (developed on 3.12). Everything runs on a laptop CPU; a GPU is used if present.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Getting the data

The dataset is [WM-811K (LSWMD.pkl)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) on Kaggle, about 2 GB.

1. Create a Kaggle API token: kaggle.com -> Settings -> API -> **Create New Token**. Save the downloaded
   `kaggle.json` to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`). Keep it out of the repo.
2. Download and prepare the data:

```bash
python scripts/download_data.py   # -> data/raw/LSWMD.pkl
python scripts/prepare_data.py    # -> data/processed/wm811k_64.npz (+ checks splits/splits_seed42.npz)
python scripts/eda.py             # -> outputs/eda/ (figures + summary.json)
pytest
```

`LSWMD.pkl` was saved with a very old pandas version that modern pandas cannot always read. `waferwise.data.read_lswmd`
tries `pandas.read_pickle` first and otherwise falls back to a small compatibility reader: it unpickles with pandas
classes replaced by inert stubs and rebuilds the DataFrame from the raw NumPy column blocks, so it works with any pandas
version. That reader only allows NumPy and a few safe built-ins to be loaded from the file.

### Data processing choices

- Only the 172,950 labelled wafers are used (9 classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none).
- Maps are resized to 64x64 with nearest-neighbour sampling, so pixels stay valid codes (0 off-wafer, 1 pass, 2 fail).
- One stratified 70/15/15 train/val/test split (seed 42) is committed in `splits/` as LSWMD row numbers. The test set is
  never cleaned, augmented or used for tuning.
- WM-811K contains 3,250 groups of pixel-identical wafer maps. A plain random split put 704 test wafers' exact copies
  in the training set, so the split keeps every duplicate group inside a single split.

## Repository layout

```
configs/      YAML configs (no hard-coded settings)
src/waferwise/
  data.py     loading, preprocessing, splits, Dataset
  models.py   model definitions
  train.py    training loop
  evaluate.py macro-F1, per-class recall, confusion matrix
  utils.py    seeding, logging, paths
scripts/      command-line entry points
splits/       the fixed train/val/test split
cpp/          C++ inference (OpenVINO + OpenCV)
tests/        sanity tests
notebooks/    exploration only
```
