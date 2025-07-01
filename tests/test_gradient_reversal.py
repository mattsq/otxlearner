from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.models.domain import GradientReversal


def test_gradient_reversal_backward() -> None:
    layer = GradientReversal()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = layer(x, 0.5)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-0.5, -0.5, -0.5]))
    assert torch.allclose(y, x)
