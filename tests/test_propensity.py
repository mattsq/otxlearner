from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import cross_fit_propensity


def test_crossfit_deterministic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 5))
    t = rng.integers(0, 2, size=100).astype(float)
    e1 = cross_fit_propensity(x, t, n_splits=4, seed=42)
    e2 = cross_fit_propensity(x, t, n_splits=4, seed=42)
    assert np.allclose(e1, e2)
