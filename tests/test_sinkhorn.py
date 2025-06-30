from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.models.sinkhorn import Sinkhorn


def test_sinkhorn_forward_shape() -> None:
    layer = Sinkhorn(blur=0.1, p=2)
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(6, 3, requires_grad=True)
    loss = layer(x, y)
    assert loss.dim() == 0


def test_sinkhorn_backward() -> None:
    layer = Sinkhorn()
    x = torch.randn(4, 2, requires_grad=True)
    y = torch.randn(4, 2, requires_grad=True)
    loss = layer(x, y)
    loss.backward()
    assert x.grad is not None and y.grad is not None
