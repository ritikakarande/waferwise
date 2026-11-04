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

## Label audit

The labels themselves are noisy: the dataset contains pixel-identical wafers with different labels. To find likely
label errors in the **training set** (val and test labels are never changed):

1. **Out-of-sample predictions.** The training split is cut into 5 folds (duplicates kept together). For each fold a
   model is trained on the other 4 and predicts the held-out one, so every training wafer gets a prediction from a
   model that never saw it. This is done for three different models: the baseline CNN, a smaller CNN with another
   seed, and gradient-boosted trees on hand-made features (failure density grid, radial ring profile).
2. **Confident learning** ([cleanlab](https://github.com/cleanlab/cleanlab)) on the averaged predictions estimates how
   often each class is confused with each other class and flags wafers whose given label is unlikely.
3. **Ensemble disagreement.** A wafer is dropped only if cleanlab flags it **and** all three models agree on the same
   other class.

```bash
python scripts/cross_val.py --config configs/audit_cnn_a.yaml   # out-of-fold CNN predictions (folds can run in parallel)
python scripts/cross_val.py --config configs/audit_cnn_b.yaml
python scripts/label_audit.py                                   # -> audit/suspect_labels.csv + review grids
python scripts/train.py --config configs/cleaned.yaml           # retrain without the dropped wafers
```

**What it found:** 862 wafers flagged by cleanlab and 563 (0.47% of training) flagged by both signals. The suspects
concentrate in the confusable classes: 5.4% of Scratch labels, 4.8% of Loc and 2.3% of Edge-Loc, against 0.23% of
"none". The most common corrections are none -> Edge-Loc (129), none -> Loc (77), Loc -> Edge-Loc (50) and
Edge-Ring -> Edge-Loc (48). The full list with scores is in [`audit/suspect_labels.csv`](audit/suspect_labels.csv).

![Most suspicious training labels](docs/figures/label_audit_top_suspects.png)

Reviewing the grids by hand: many suspects are clear errors, such as "Center" wafers with no centre cluster, or
"none" wafers covered in random failures. Others are honest borderline cases (a partial ring between Edge-Ring and
Edge-Loc). A few are model mistakes, mostly thin scratches the 64x64 CNN does not see. So the audit is a useful filter
for a human reviewer, not a replacement for one.

**Effect of removing the 563 suspects** (mean +/- std over 2 seeds each):

| Train data | Val macro-F1 | Test macro-F1 | Test macro recall | Test macro precision | Test recall Scratch / Center / Random |
|---|---|---|---|---|---|
| Real, uncleaned | 0.871 +/- 0.003 | 0.871 +/- 0.001 | 0.882 | 0.864 | 0.70 / 0.91 / 0.94 |
| Real, cleaned | 0.869 +/- 0.002 | 0.869 +/- 0.002 | 0.892 | 0.849 | 0.74 / 0.93 / 0.96 |

Removing 0.5% of the labels did **not** change macro-F1 beyond seed-to-seed noise. It shifted the model slightly
towards finding more defects (recall +0.01) at the cost of a few more false alarms (precision -0.015). Two honest
caveats: the gain is within noise for every class, and val/test contain the same kind of label errors (they are
deliberately left untouched), so a model that has stopped learning "defect-looking wafer = none" is partly penalised
by those noisy labels. The cleaned training set is carried forward to the synthetic-data experiments.

## Synthetic rare defects

The four rarest classes have only 107 (Near-full), 384 (Donut), 598 (Random) and 791 (Scratch) cleaned training
wafers. A small **conditional VAE** (1.8M parameters, 12 minutes on a laptop CPU) learns where failing dies appear for
each class and generates new ones:

- It models only the *fail map* (which dies failed), conditioned on the class. Training uses balanced epochs (every
  class seen equally often) and random rotations/flips, on the cleaned training split only.
- Each synthetic wafer borrows its outline and die grid from a random real wafer of the same class. Dies pass or fail
  **one die at a time**, using the model's average failure probability over that die. Sampling every pixel
  independently gave fine salt-and-pepper noise that looked nothing like real maps (most wafers have only ~25x27 dies
  scaled up to 64x64); sampling per die reproduces their blocky look.

```bash
python scripts/train_vae.py            # -> outputs/cvae/model.pt
python scripts/generate_synthetic.py   # -> data/synthetic/cvae.npz (2,000 per rare class)
python scripts/realism_check.py        # -> outputs/realism/summary.json
```

![Real vs synthetic wafers](docs/figures/cvae_real_vs_synthetic.png)

**Realism check** (all on training data; val and test are not used):

| Class | Real-vs-synthetic detector AUC (CNN / trees) | Baseline classifier picks the intended class | Copies of training wafers |
|---|---|---|---|
| Near-full | 0.83 / 0.92 | 79% | none |
| Random | 0.84 / 0.87 | 85% | none |
| Donut | 0.89 / 0.95 | 53% (else mostly Loc) | none |
| Scratch | 0.89 / 0.85 | **2%** (almost all "none") | none |

The detector AUCs (0.5 would mean indistinguishable) show the synthetic wafers are still recognisably synthetic,
which is typical for a VAE: it averages over possibilities and produces softer patterns than real wafers. Copying is
ruled out: no synthetic wafer is closer to a training wafer than the closest 5% of real-to-real pairs. Blob- and
texture-like defects (Near-full, Random, partly Donut) come out recognisable. Thin scratches do not: the VAE blurs a
one-die-wide line into background noise. Sharpening the probabilities or lowering the sampling temperature helped
Near-full and Donut a little but hurt Random, and did not rescue Scratch, so the plain settings are kept.

Whether these samples actually improve rare-class recall is tested next, against simple oversampling of the real
rare wafers.

## Repository layout

```
configs/      YAML configs (no hard-coded settings)
src/waferwise/
  data.py     loading, preprocessing, splits, Dataset
  models.py   model definitions
  train.py    training loop
  evaluate.py macro-F1, per-class recall, confusion matrix
  label_audit.py  cross-validation folds, cleanlab + ensemble disagreement
  generative.py   conditional VAE and die-level sampling of synthetic wafers
  utils.py    seeding, logging, paths
scripts/      command-line entry points
splits/       the fixed train/val/test split
audit/        suspect-label list from the label audit (defines the cleaned training set)
cpp/          C++ inference (OpenVINO + OpenCV)
tests/        sanity tests
notebooks/    exploration only
```
