from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import KFold  # type: ignore[import-untyped]


Array = npt.NDArray[np.float64]

if TYPE_CHECKING:  # pragma: no cover - type hints
    from ..data import IHDPDataset


def crossfit_propensity(
    x: Array,
    t: Array,
    *,
    folds: int = 5,
    seed: int = 0,
    clip: float | None = None,
) -> Array:
    """Estimate propensities via K-fold cross-fitted logistic regression."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.empty_like(t, dtype=np.float64)
    for train_idx, val_idx in kf.split(x):
        model = LogisticRegression(max_iter=1000)
        model.fit(x[train_idx], t[train_idx])
        preds[val_idx] = model.predict_proba(x[val_idx])[:, 1]
    if clip is not None:
        preds = np.clip(preds, clip, 1 - clip)
    return preds


def estimate_propensity_splits(
    ds: "IHDPDataset",
    *,
    folds: int = 5,
    seed: int = 0,
    epsilon_prop: float = 0.05,
) -> Tuple[Array, Array, Array]:
    """Return clipped cross-fitted propensities for train/val/test."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x_train_val = np.concatenate([ds.train.x, ds.val.x])
    t_train_val = np.concatenate([ds.train.t, ds.val.t])

    cf_pred = crossfit_propensity(
        x_train_val, t_train_val, folds=folds, seed=seed, clip=epsilon_prop
    )
    n_train = ds.train.x.shape[0]
    p_train = cf_pred[:n_train]
    p_val = cf_pred[n_train:]

    model_full = LogisticRegression(max_iter=1000)
    model_full.fit(x_train_val, t_train_val)
    p_test = model_full.predict_proba(ds.test.x)[:, 1]
    p_test = np.clip(p_test, epsilon_prop, 1 - epsilon_prop)
    return p_train, p_val, p_test
