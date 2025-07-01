from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Simple MLP encoder with outcome and tau heads."""

    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.GELU())
            in_dim = size

        self.net = nn.Sequential(*layers)
        self.outcome_head = nn.Linear(in_dim, 1)
        self.tau_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.net(x)
        outcome = self.outcome_head(feats).squeeze(-1)
        tau = self.tau_head(feats).squeeze(-1)
        return outcome, tau


__all__ = ["MLPEncoder"]
