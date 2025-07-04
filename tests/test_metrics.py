from __future__ import annotations

import torch

from otxlearner.utils.metrics import ate, pehe, policy_risk


def test_pehe() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    true = torch.tensor([1.0, 1.0, 1.0])
    expected = torch.sqrt(torch.mean((pred - true) ** 2)).item()
    assert abs(pehe(pred, true) - expected) < 1e-6


def test_ate() -> None:
    pred = torch.tensor([0.0, 2.0, 4.0])
    true = torch.tensor([0.0, 1.0, 3.0])
    expected = pred.mean().item() - true.mean().item()
    assert abs(ate(pred, true) - expected) < 1e-6


def test_policy_risk() -> None:
    tau_pred = torch.tensor([1.0, -1.0])
    tau_true = torch.tensor([1.0, -1.0])
    mu0 = torch.tensor([0.0, 0.0])
    mu1 = torch.tensor([1.0, 1.0])
    risk = policy_risk(tau_pred, tau_true, mu0, mu1)
    assert abs(risk) < 1e-6
