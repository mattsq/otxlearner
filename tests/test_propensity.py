from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.propensity import crossfit_propensity


def test_crossfit_deterministic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 5))
    t = rng.integers(0, 2, size=100)
    p1 = crossfit_propensity(x, t, folds=3, seed=42)
    p2 = crossfit_propensity(x, t, folds=3, seed=42)
    assert np.allclose(p1, p2)
