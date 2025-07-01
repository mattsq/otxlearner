from __future__ import annotations

import numpy as np
from pathlib import Path
from otxlearner.data import load_twins


def _create_fake_dataset(path: Path) -> None:
    n, d = 100, 5
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, d))
    t = rng.integers(0, 2, size=n)
    y0 = rng.normal(size=n)
    y1 = rng.normal(size=n)
    np.savez(path / "twins.npz", x=x, t=t, y0=y0, y1=y1)


def test_twins_deterministic(tmp_path: Path) -> None:
    _create_fake_dataset(tmp_path)
    ds1 = load_twins(tmp_path, val_fraction=0.2, seed=0)
    ds2 = load_twins(tmp_path, val_fraction=0.2, seed=0)
    assert np.array_equal(ds1.train.x, ds2.train.x)
    assert np.array_equal(ds1.val.x, ds2.val.x)
    assert np.array_equal(ds1.test.x, ds2.test.x)


def test_twins_split_shapes(tmp_path: Path) -> None:
    _create_fake_dataset(tmp_path)
    ds = load_twins(tmp_path, val_fraction=0.2, seed=1)
    n = 100
    val_size = int(n * 0.2)
    test_size = val_size
    train_size = n - val_size - test_size
    assert ds.train.x.shape == (train_size, 5)
    assert ds.val.x.shape == (val_size, 5)
    assert ds.test.x.shape == (test_size, 5)
