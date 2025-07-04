from __future__ import annotations

from typing import cast

import torch
from torch import nn


class _MLP(nn.Module):
    """Simple 2-layer MLP used in RealNVP coupling blocks."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return cast(torch.Tensor, self.net(x))


class Coupling(nn.Module):
    """RealNVP affine coupling layer."""

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.mask: torch.Tensor
        self.register_buffer("mask", mask)
        self.scale_net = _MLP(dim, hidden_dim)
        self.translate_net = _MLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        x_masked = x * self.mask
        s = cast(torch.Tensor, self.scale_net(x_masked)) * (1 - self.mask)
        t = cast(torch.Tensor, self.translate_net(x_masked)) * (1 - self.mask)
        if reverse:
            out = x_masked + (1 - self.mask) * (x - t) * torch.exp(-s)
        else:
            out = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        return cast(torch.Tensor, out)


class RealNVP(nn.Module):
    """Sequence of RealNVP coupling layers."""

    def __init__(self, dim: int, hidden_dim: int = 64, n_flows: int = 2) -> None:
        super().__init__()
        masks = [self._create_mask(dim, i % 2 == 0) for i in range(n_flows)]
        self.layers = nn.ModuleList([Coupling(dim, hidden_dim, m) for m in masks])

    @staticmethod
    def _create_mask(dim: int, even: bool) -> torch.Tensor:
        mask = torch.zeros(dim)
        mask[::2] = 1 if even else 0
        mask[1::2] = 0 if even else 1
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        for layer in reversed(self.layers):
            out = layer(out, reverse=True)
        return out


class FlowEncoder(nn.Module):
    """Encoder with a RealNVP flow followed by outcome and tau heads."""

    def __init__(
        self, input_dim: int, *, n_flows: int = 2, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.flow: RealNVP = RealNVP(input_dim, hidden_dim, n_flows)
        self.outcome_head = nn.Linear(input_dim, 1)
        self.tau_head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.flow(x)
        outcome = self.outcome_head(feats).squeeze(-1)
        tau = self.tau_head(feats).squeeze(-1)
        return outcome, tau

    def predict_tau(self, x: torch.Tensor) -> torch.Tensor:
        """Return treatment effect predictions for ``x``."""
        _outcome, tau = self.forward(x)
        return tau

    def inverse(
        self, z: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - simple inverse
        return self.flow.inverse(z)


__all__ = ["FlowEncoder", "RealNVP"]
