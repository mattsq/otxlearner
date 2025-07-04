import os
from pathlib import Path

import numpy as np
import pytest

from otxlearner.data.ihdp import IHDPDataset, IHDPSplit
from otxlearner.registries import _DATASETS


@pytest.fixture()
def ihdp_root(tmp_path: Path) -> Path:
    root_env = os.getenv("IHDP_DATA")
    if root_env:
        path = Path(root_env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path


@pytest.fixture()
def fast_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch dataset and propensity estimation for fast smoke tests."""

    rng = np.random.default_rng(0)

    def make_split() -> IHDPSplit:
        n, d = 20, 5
        return IHDPSplit(
            x=rng.normal(size=(n, d)),
            t=rng.integers(0, 2, size=n).astype(np.float64),
            yf=rng.normal(size=n),
            ycf=rng.normal(size=n),
            mu0=rng.normal(size=n),
            mu1=rng.normal(size=n),
        )

    ds = IHDPDataset(make_split(), make_split(), make_split())
    monkeypatch.setitem(_DATASETS, "ihdp", lambda root=None: ds)
    monkeypatch.setattr(
        "otxlearner.cli.cross_fit_propensity",
        lambda x, t, *, n_splits=5, seed=0, clip=1e-3: np.full_like(t, 0.5),
    )

    yield
