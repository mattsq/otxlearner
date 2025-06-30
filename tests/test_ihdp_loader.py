from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import load_ihdp


def test_ihdp_deterministic(tmp_path) -> None:
    ds1 = load_ihdp(tmp_path, val_fraction=0.2, seed=0)
    ds2 = load_ihdp(tmp_path, val_fraction=0.2, seed=0)
    assert np.array_equal(ds1.train.x, ds2.train.x)
    assert np.array_equal(ds1.val.x, ds2.val.x)


def test_ihdp_split_sizes(tmp_path) -> None:
    ds = load_ihdp(tmp_path, val_fraction=0.1, seed=1)
    n_total = 67200
    assert ds.train.x.shape[0] + ds.val.x.shape[0] == n_total
    assert ds.test.x.shape[0] == 7500
