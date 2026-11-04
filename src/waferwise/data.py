"""Loading, preprocessing and splitting the WM-811K wafer-map dataset.

Pipeline (run once via ``scripts/prepare_data.py``):
    LSWMD.pkl -> keep labelled wafers -> resize to SxS (nearest neighbour)
    -> data/processed/wm811k_<S>.npz  -> fixed stratified train/val/test split.

Every later experiment loads the processed .npz and the split file, so all
results are computed on exactly the same samples.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from waferwise.utils import get_logger, repo_path

CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# Pixel values in a wafer map: 0 = off-wafer, 1 = passing die, 2 = failing die.
VALID_VALUES = (0, 1, 2)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reading the raw pickle
# ---------------------------------------------------------------------------
#
# LSWMD.pkl was written by a very old pandas (0.x). Modern pandas cannot always
# unpickle it: the pickle refers to modules that no longer exist (e.g.
# ``pandas.indexes``) and to an old internal BlockManager layout.
#
# Fallback strategy: unpickle with every pandas class replaced by a tiny stub
# that only records its constructor arguments and state. The pickle then loads
# with any pandas version, and we rebuild the DataFrame ourselves from the raw
# column blocks (plain NumPy arrays). Nothing in the file is executed besides
# pandas/NumPy reconstruction calls.


class _PandasStub:
    """Stand-in for a pandas object: remembers constructor args and pickled state."""

    pandas_name = "?"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.args = args
        obj.state = None
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _LegacyUnpickler(pickle.Unpickler):
    _stubs: dict[str, type] = {}
    # Only these non-pandas modules may be loaded; anything else is rejected.
    _allowed_modules = ("numpy", "copyreg", "copy_reg", "builtins", "__builtin__", "collections", "_codecs")

    def find_class(self, module, name):
        if module == "pandas" or module.startswith("pandas."):
            key = f"{module}.{name}"
            if key not in self._stubs:
                self._stubs[key] = type(name, (_PandasStub,), {"pandas_name": key})
            return self._stubs[key]
        is_builtin = module in ("builtins", "__builtin__")
        safe_builtin = name in ("object", "slice", "range", "set", "frozenset", "complex", "bytearray")
        if module.split(".")[0] not in self._allowed_modules or (is_builtin and not safe_builtin):
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from pickle")
        return super().find_class(module, name)


def _index_values(index) -> np.ndarray:
    """Turn a stubbed pandas Index back into a NumPy array of labels."""
    if isinstance(index, np.ndarray):
        return index
    if isinstance(index, _PandasStub):
        # Indexes are pickled as ``_new_Index(cls, attributes_dict)``.
        attrs = index.args[1] if len(index.args) == 2 and isinstance(index.args[1], dict) else None
        if attrs is None and isinstance(index.state, dict):
            attrs = index.state
        if attrs is not None:
            if "data" in attrs:
                return np.asarray(attrs["data"])
            if "start" in attrs:  # RangeIndex
                return np.arange(attrs["start"], attrs["stop"], attrs.get("step", 1))
    raise ValueError(f"Cannot interpret pickled index: {getattr(index, 'pandas_name', type(index))}")


def _rebuild_dataframe(frame: _PandasStub) -> pd.DataFrame:
    """Rebuild a DataFrame from the stubbed BlockManager state of an old pickle."""
    manager = frame.state["_data"] if isinstance(frame.state, dict) else frame.state
    state = manager.state
    extra = state[3] if isinstance(state, tuple) and len(state) == 4 else None
    if not (isinstance(extra, dict) and "0.14.1" in extra):
        raise ValueError("Unrecognised legacy DataFrame layout in pickle")

    layout = extra["0.14.1"]
    columns = _index_values(layout["axes"][0])
    index = _index_values(layout["axes"][1])
    data = {}
    for block in layout["blocks"]:
        positions = np.arange(len(columns))[block["mgr_locs"]]
        values = np.asarray(block["values"])
        for row, pos in enumerate(positions):
            data[str(columns[pos])] = values[row]
    return pd.DataFrame({str(c): data[str(c)] for c in columns}, index=index)


def read_lswmd(path: str | Path) -> pd.DataFrame:
    """Load LSWMD.pkl, falling back to the legacy reader if modern pandas fails."""
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` "
            "(see README, 'Getting the data')."
        )
    try:
        df = pd.read_pickle(path)
        log.info("Loaded %s with pandas.read_pickle", path.name)
        return df
    except Exception as err:  # old pickle formats raise many different errors
        log.info("pandas.read_pickle failed (%s: %s); using legacy reader", type(err).__name__, err)

    with open(path, "rb") as f:
        obj = _LegacyUnpickler(f, encoding="latin1").load()
    df = _rebuild_dataframe(obj)
    log.info("Loaded %s with legacy reader", path.name)
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def flatten_label(raw) -> str | None:
    """``[['Center']]`` -> ``'Center'``; empty arrays (unlabelled) -> ``None``."""
    arr = np.asarray(raw, dtype=object)
    if arr.size == 0:
        return None
    return str(arr.ravel()[0]).strip()


def nearest_index(n_in: int, n_out: int) -> np.ndarray:
    """For each of ``n_out`` output pixels, the input index whose centre it falls on."""
    return ((np.arange(n_out) + 0.5) * n_in / n_out).astype(int)


def resize_nearest(wafer: np.ndarray, size: int) -> np.ndarray:
    """Resize a categorical wafer map with nearest-neighbour sampling.

    Each output pixel copies the input pixel at its centre, so the result only
    ever contains values that were in the input (no blending of 0/1/2 codes).
    """
    h, w = wafer.shape
    return wafer[np.ix_(nearest_index(h, size), nearest_index(w, size))]


def wafer_hash(wafer: np.ndarray) -> int:
    """64-bit fingerprint of an original wafer map, used to find exact duplicates."""
    wafer = np.ascontiguousarray(wafer, dtype=np.uint8)
    digest = hashlib.blake2b(wafer.tobytes() + str(wafer.shape).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little")


def preprocess(df: pd.DataFrame, size: int) -> dict[str, np.ndarray]:
    """Keep labelled wafers, resize them and collect per-wafer metadata."""
    labels = df["failureType"].map(flatten_label)
    keep = labels.isin(CLASSES).to_numpy()
    log.info("Labelled wafers: %d of %d", keep.sum(), len(df))

    rows = np.flatnonzero(keep)
    n = len(rows)
    images = np.empty((n, size, size), dtype=np.uint8)
    orig_shape = np.empty((n, 2), dtype=np.int32)
    hashes = np.empty(n, dtype=np.uint64)
    n_invalid = 0
    for i, row in enumerate(rows):
        wafer = np.asarray(df["waferMap"].iat[row])
        if not np.isin(wafer, VALID_VALUES).all():
            n_invalid += 1
        images[i] = resize_nearest(wafer, size)
        orig_shape[i] = wafer.shape
        hashes[i] = wafer_hash(wafer)
    if n_invalid:
        raise ValueError(f"{n_invalid} wafer maps contain values outside {VALID_VALUES}")

    split_col = "trianTestLabel" if "trianTestLabel" in df else "trainTestLabel"
    return {
        "X": images,
        "y": labels.iloc[rows].map(CLASS_TO_IDX).to_numpy(dtype=np.int64),
        "classes": np.array(CLASSES),
        "source_index": rows.astype(np.int64),  # row number in LSWMD.pkl
        "orig_shape": orig_shape,
        "map_hash": hashes,
        "lot_name": np.asarray(df["lotName"].iloc[rows].astype(str).tolist(), dtype=str),
        "wafer_index": df["waferIndex"].iloc[rows].to_numpy(dtype=np.float64).astype(np.int32),
        "die_size": df["dieSize"].iloc[rows].to_numpy(dtype=np.float32),
        "orig_split": np.asarray([flatten_label(v) or "" for v in df[split_col].iloc[rows]], dtype=str),
    }


def save_processed(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    # Object arrays would need pickle to load again; keep the file pickle-free.
    object_keys = [key for key, arr in arrays.items() if arr.dtype == object]
    if object_keys:
        raise TypeError(f"Arrays with object dtype cannot be saved: {object_keys}")
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_processed(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(repo_path(path), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def make_splits(
    y: np.ndarray,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Stratified train/val/test split; returns sorted positions into ``y``.

    If ``groups`` is given (e.g. a hash of each wafer map), all members of a group
    land in the same split, so exact duplicates cannot leak from train into test.
    Groups are stratified by the label of their first member.
    """
    if groups is None:
        groups = np.arange(len(y))
    _, first, group_id = np.unique(groups, return_index=True, return_inverse=True)

    reps = np.arange(len(first))  # one representative per group
    rep_y = y[first]
    train_val, test = train_test_split(reps, test_size=test_fraction, stratify=rep_y, random_state=seed)
    val_share = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(train_val, test_size=val_share, stratify=rep_y[train_val], random_state=seed)

    split_of_group = np.empty(len(first), dtype=np.int8)
    split_of_group[train], split_of_group[val], split_of_group[test] = 0, 1, 2
    split_of_sample = split_of_group[group_id.ravel()]
    return {name: np.flatnonzero(split_of_sample == k) for k, name in enumerate(("train", "val", "test"))}


def save_splits(splits: dict[str, np.ndarray], source_index: np.ndarray, path: str | Path) -> None:
    """Store splits as LSWMD row numbers so they do not depend on array order."""
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: source_index[pos] for name, pos in splits.items()})


def load_splits(path: str | Path, source_index: np.ndarray) -> dict[str, np.ndarray]:
    """Load a split file and map LSWMD row numbers back to positions in the processed arrays."""
    with np.load(repo_path(path)) as npz:
        out = {}
        for name in ("train", "val", "test"):
            pos = np.searchsorted(source_index, npz[name])
            if not np.array_equal(source_index[pos], npz[name]):
                raise ValueError(f"Split '{name}' refers to wafers missing from the processed data")
            out[name] = pos
    return out


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------


class WaferDataset(Dataset):
    """Wafer maps as 1x S x S float tensors scaled to [0, 1] (0 off-wafer, 0.5 pass, 1 fail)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None, transform=None):
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = torch.from_numpy(self.images[idx].astype(np.float32) / 2.0).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[idx])
