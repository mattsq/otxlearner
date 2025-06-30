from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import ate, pehe


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
