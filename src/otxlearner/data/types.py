from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Protocol

__all__ = ["SplitProtocol", "DatasetProtocol"]


class SplitProtocol(Protocol):
    x: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    yf: npt.NDArray[np.float64]
    ycf: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    mu1: npt.NDArray[np.float64]


class DatasetProtocol(Protocol):
    train: SplitProtocol
    val: SplitProtocol
    test: SplitProtocol
