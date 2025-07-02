from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import torch

from .types import DatasetProtocol, SplitProtocol

ArrayF = npt.NDArray[np.float64]


@dataclass
class TorchSplit(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """Torch dataset wrapper around :class:`IHDPSplit`."""

    x: torch.Tensor
    t: torch.Tensor
    yf: torch.Tensor
    mu0: torch.Tensor
    mu1: torch.Tensor
    e: torch.Tensor

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.x[idx],
            self.t[idx],
            self.yf[idx],
            self.mu0[idx],
            self.mu1[idx],
            self.e[idx],
        )


@dataclass
class TorchIHDP:
    train: TorchSplit
    val: TorchSplit
    test: TorchSplit


def _to_tensor(x: torch.Tensor | ArrayF) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def torchify(
    ds: DatasetProtocol,
    propensities: tuple[ArrayF, ArrayF, ArrayF] | None = None,
) -> TorchIHDP:
    """Convert numpy dataset to PyTorch tensors."""

    def convert(split: SplitProtocol, e: ArrayF) -> TorchSplit:
        return TorchSplit(
            _to_tensor(split.x),
            _to_tensor(split.t),
            _to_tensor(split.yf),
            _to_tensor(split.mu0),
            _to_tensor(split.mu1),
            _to_tensor(e),
        )

    if propensities is None:
        zeros_train = np.zeros(len(ds.train.x), dtype=np.float64)
        zeros_val = np.zeros(len(ds.val.x), dtype=np.float64)
        zeros_test = np.zeros(len(ds.test.x), dtype=np.float64)
        propensities = (zeros_train, zeros_val, zeros_test)

    e_train, e_val, e_test = propensities
    return TorchIHDP(
        convert(ds.train, e_train),
        convert(ds.val, e_val),
        convert(ds.test, e_test),
    )


__all__ = ["TorchIHDP", "TorchSplit", "_to_tensor", "torchify"]
