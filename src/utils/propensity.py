from __future__ import annotations


import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import KFold  # type: ignore[import-untyped]


ArrayF = npt.NDArray[np.float64]


def cross_fit_propensity(
    x: ArrayF, t: ArrayF, *, n_splits: int = 5, seed: int = 0, clip: float = 1e-3
) -> ArrayF:
    """Estimate propensities with K-fold cross-fitting."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    e_hat = np.zeros_like(t, dtype=np.float64)
    for train_idx, test_idx in kf.split(x):
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(x[train_idx], t[train_idx])
        e_hat[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return np.clip(e_hat, clip, 1.0 - clip)


__all__ = ["cross_fit_propensity"]
