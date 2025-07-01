from __future__ import annotations

import torch

from otxlearner.models.unbalanced_sinkhorn import UnbalancedSinkhorn


def test_unbalanced_sinkhorn_forward_shape() -> None:
    layer = UnbalancedSinkhorn(blur=0.1)
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(6, 3, requires_grad=True)
    loss = layer(x, y)
    assert loss.dim() == 0


def test_unbalanced_sinkhorn_backward() -> None:
    layer = UnbalancedSinkhorn()
    x = torch.randn(4, 2, requires_grad=True)
    y = torch.randn(4, 2, requires_grad=True)
    loss = layer(x, y)
    loss.backward()
    assert x.grad is not None and y.grad is not None
