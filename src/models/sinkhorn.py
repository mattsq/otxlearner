"""Sinkhorn layer."""

from __future__ import annotations

from typing import Callable

import torch
from geomloss import SamplesLoss


class Sinkhorn(torch.nn.Module):
    """Compute Sinkhorn divergence between two point clouds."""

    def __init__(self, *, blur: float = 0.05, p: int = 2) -> None:
        super().__init__()
        self.blur = blur
        self.p = p
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Inputs must be 2D matrices")
        return self.loss_fn(x, y)
