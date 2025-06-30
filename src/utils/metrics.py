from __future__ import annotations

import torch


def pehe(tau_pred: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Return Precision in Estimation of Heterogeneous Effect."""
    return float(torch.sqrt(torch.mean((tau_pred - tau_true) ** 2)).item())


def ate(tau_pred: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Return Average Treatment Effect error."""
    return float(tau_pred.mean().item() - tau_true.mean().item())


__all__ = ["pehe", "ate"]
