"""Training and validation helpers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data.torch_adapter import TorchIHDP


def prepare_loaders(
    ds: TorchIHDP, batch_size: int, *, seed: int = 0
) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]:
    """Return train/validation loaders with fixed seed."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        ds.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(ds.val, batch_size=batch_size)
    return train_loader, val_loader


__all__ = ["prepare_loaders"]
