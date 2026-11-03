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

## Training and evaluation

```bash
python scripts/train.py --config configs/baseline.yaml           # -> outputs/baseline/
python scripts/evaluate.py --run outputs/baseline --split test   # final numbers only, never for tuning
```

Each run writes `model.pt`, `history.json`, `training_curves.png`, `metrics_val.json` and a row-normalised
confusion matrix (`confusion_val.png`) to `outputs/<experiment_name>/`. Models are selected by validation macro-F1.
Every report includes macro-F1 and per-class recall, because plain accuracy is misleading here: predicting "none" for
every wafer already scores 85%.

**Baseline:** a 4-block CNN (~99k parameters) on 1x64x64 maps, AdamW, cosine learning rate, random flips and
90-degree rotations, and cross-entropy weighted by 1/sqrt(class frequency).

## Results so far

| Model | Train data | Val macro-F1 | Test macro-F1 | Test macro recall | Test accuracy |
|---|---|---|---|---|---|
| Baseline CNN (FP32, PyTorch) | real, uncleaned | 0.873 | 0.871 | 0.893 | 0.972 |

Per-class test recall for the baseline:

| Center | Donut | Edge-Loc | Edge-Ring | Loc | Near-full | Random | Scratch | none |
|---|---|---|---|---|---|---|---|---|
| 0.93 | 0.89 | 0.85 | 0.97 | 0.74 | 0.95 | 0.95 | 0.77 | 0.99 |

The weakest classes are Loc and Scratch. 13% of each is predicted as "none", and in the other direction "none" wafers
are most often confused with Edge-Loc, Loc, Center and Scratch. Small local defects and clean wafers are the hardest
boundary, and that is where the label audit looks next.

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
