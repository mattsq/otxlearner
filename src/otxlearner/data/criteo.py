from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import gzip

import numpy as np
import numpy.typing as npt

from .utils import download

__all__ = ["CriteoSplit", "CriteoDataset", "load_criteo_uplift"]

URL = (
    "https://storage.googleapis.com/recorder-public/criteo-research-uplift-v2.1.csv.gz"
)
SHA = ""  # checksum omitted for brevity


@dataclass
class CriteoSplit:
    x: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    yf: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    mu1: npt.NDArray[np.float64]


@dataclass
class CriteoDataset:
    train: CriteoSplit
    val: CriteoSplit
    test: CriteoSplit


def _load_csv(path: Path, nrows: int | None) -> dict[str, npt.NDArray[np.float64]]:
    with gzip.open(path, "rt") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for i, row in enumerate(reader):
            rows.append(row)
            if nrows is not None and i + 1 >= nrows:
                break
    cols = rows[0].keys()
    data: dict[str, list[float]] = {c: [] for c in cols}
    for row in rows:
        for k, v in row.items():
            data[k].append(float(v))
    return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}


def load_criteo_uplift(
    root: str | Path = Path.home() / ".cache" / "otxlearner" / "criteo",
    *,
    validate: bool | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    nrows: int | None = None,
) -> CriteoDataset:
    """Load the public Criteo Uplift dataset (small subset by default)."""
    root = Path(root)
    path = root / "criteo.csv.gz"
    if validate is None:
        validate = root == Path.home() / ".cache" / "otxlearner" / "criteo"
    if not path.exists() or validate:
        download(URL, path, sha256=SHA)
    data = _load_csv(path, nrows)
    feature_names = [
        c for c in data.keys() if c not in {"treatment", "conversion", "visit"}
    ]
    x = np.stack([data[c] for c in feature_names], axis=1)
    t = data["treatment"]
    y = data["conversion"]
    mu0 = y.copy()
    mu1 = y.copy()

    rng = np.random.default_rng(42)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)

    val_size = int(len(idx) * val_fraction)
    test_size = int(len(idx) * test_fraction)
    train_idx = idx[: -val_size - test_size]
    val_idx = idx[-val_size - test_size : -test_size]
    test_idx = idx[-test_size:]

    def split(i: npt.NDArray[np.int_]) -> CriteoSplit:
        return CriteoSplit(x=x[i], t=t[i], yf=y[i], mu0=mu0[i], mu1=mu1[i])

    train = split(train_idx)
    val = split(val_idx)
    test = split(test_idx)
    return CriteoDataset(train=train, val=val, test=test)
