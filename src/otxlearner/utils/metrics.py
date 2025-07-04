from __future__ import annotations

import torch


def pehe(tau_pred: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Return Precision in Estimation of Heterogeneous Effect."""
    return float(torch.sqrt(torch.mean((tau_pred - tau_true) ** 2)).item())


def ate(tau_pred: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Return Average Treatment Effect error."""
    return float(tau_pred.mean().item() - tau_true.mean().item())


def policy_risk(
    tau_pred: torch.Tensor,
    tau_true: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
) -> float:
    """Return policy risk w.r.t. sign of τ."""
    best_value = torch.where(tau_true > 0, mu1, mu0).mean().item()
    pred_value = torch.where(tau_pred > 0, mu1, mu0).mean().item()
    return float(best_value - pred_value)


__all__ = ["pehe", "ate", "policy_risk"]
