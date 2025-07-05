from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from otxlearner.data import load_tabular


def _create_df(n: int = 50, d: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"f{i}": rng.normal(size=n) for i in range(d)}
    data["t"] = rng.integers(0, 2, size=n)
    data["y"] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_load_tabular_dataframe_deterministic() -> None:
    df = _create_df()
    ds1 = load_tabular(
        df,
        features=["f0", "f1", "f2"],
        treatment="t",
        outcome="y",
        val_fraction=0.2,
        test_fraction=0.1,
        seed=0,
    )
    ds2 = load_tabular(
        df,
        features=["f0", "f1", "f2"],
        treatment="t",
        outcome="y",
        val_fraction=0.2,
        test_fraction=0.1,
        seed=0,
    )
    assert np.array_equal(ds1.train.x, ds2.train.x)
    assert np.array_equal(ds1.val.x, ds2.val.x)
    assert np.array_equal(ds1.test.x, ds2.test.x)


def test_load_tabular_csv(tmp_path: Path) -> None:
    df = _create_df()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    ds = load_tabular(
        csv_path,
        features=["f0", "f1", "f2"],
        treatment="t",
        outcome="y",
        val_fraction=0.2,
        test_fraction=0.1,
        seed=1,
    )
    n = len(df)
    val_size = int(n * 0.2)
    test_size = int(n * 0.1)
    train_size = n - val_size - test_size
    assert ds.train.x.shape == (train_size, 3)
    assert ds.val.x.shape == (val_size, 3)
    assert ds.test.x.shape == (test_size, 3)
