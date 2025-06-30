from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Simple MLP encoder with outcome and tau heads."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.outcome_head = nn.Linear(64, 1)
        self.tau_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.net(x)
        outcome = self.outcome_head(feats).squeeze(-1)
        tau = self.tau_head(feats).squeeze(-1)
        return outcome, tau


__all__ = ["MLPEncoder"]
