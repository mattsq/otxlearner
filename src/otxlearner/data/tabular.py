from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["TabularSplit", "TabularDataset", "load_tabular"]


@dataclass
class TabularSplit:
    x: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    yf: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    mu1: npt.NDArray[np.float64]


@dataclass
class TabularDataset:
    train: TabularSplit
    val: TabularSplit
    test: TabularSplit


def load_tabular(
    data: str | Path | pd.DataFrame,
    *,
    features: Sequence[str],
    treatment: str,
    outcome: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> TabularDataset:
    """Create train/val/test splits from a CSV file or DataFrame."""

    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data

    x = df.loc[:, list(features)].to_numpy(dtype=np.float64)
    t = df.loc[:, treatment].to_numpy(dtype=np.float64)
    y = df.loc[:, outcome].to_numpy(dtype=np.float64)
    mu0 = y.copy()
    mu1 = y.copy()

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    val_size = int(len(idx) * val_fraction)
    test_size = int(len(idx) * test_fraction)
    train_idx = idx[: -val_size - test_size]
    val_idx = idx[-val_size - test_size : -test_size]
    test_idx = idx[-test_size:]

    def split(i: npt.NDArray[np.int_]) -> TabularSplit:
        return TabularSplit(x=x[i], t=t[i], yf=y[i], mu0=mu0[i], mu1=mu1[i])

    train = split(train_idx)
    val = split(val_idx)
    test = split(test_idx)
    return TabularDataset(train=train, val=val, test=test)
