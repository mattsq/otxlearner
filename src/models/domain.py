from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal layer from DANN."""

    def forward(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        return cast(torch.Tensor, _GradientReversalFn.apply(x, lambd))  # type: ignore[no-untyped-call]


class DomainDiscriminator(nn.Module):
    """Simple discriminator predicting treatment domain."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin
        return cast(torch.Tensor, self.net(x))


__all__ = ["GradientReversal", "DomainDiscriminator"]
