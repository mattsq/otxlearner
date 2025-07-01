from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import load_acic


def _create_fake_dataset(path: Path) -> None:
    n, d = 120, 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, d))
    t = rng.integers(0, 2, size=n)
    y0 = rng.normal(size=n)
    y1 = rng.normal(size=n)
    np.savez(path / "acic.npz", x=x, t=t, y0=y0, y1=y1)


def test_acic_deterministic(tmp_path: Path) -> None:
    _create_fake_dataset(tmp_path)
    ds1 = load_acic(tmp_path, val_fraction=0.2, test_fraction=0.1, seed=0)
    ds2 = load_acic(tmp_path, val_fraction=0.2, test_fraction=0.1, seed=0)
    assert np.array_equal(ds1.train.x, ds2.train.x)
    assert np.array_equal(ds1.val.x, ds2.val.x)
    assert np.array_equal(ds1.test.x, ds2.test.x)


def test_acic_split_shapes(tmp_path: Path) -> None:
    _create_fake_dataset(tmp_path)
    ds = load_acic(tmp_path, val_fraction=0.2, test_fraction=0.1, seed=1)
    n = 120
    val_size = int(n * 0.2)
    test_size = int(n * 0.1)
    train_size = n - val_size - test_size
    assert ds.train.x.shape == (train_size, 4)
    assert ds.val.x.shape == (val_size, 4)
    assert ds.test.x.shape == (test_size, 4)
