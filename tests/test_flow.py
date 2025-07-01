from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.models.flow import FlowEncoder, RealNVP


def test_flow_encoder_shapes() -> None:
    model = FlowEncoder(4)
    x = torch.randn(2, 4)
    out, tau = model(x)
    assert out.shape == (2,)
    assert tau.shape == (2,)


def test_realnvp_inverse() -> None:
    flow = RealNVP(4, n_flows=2)
    x = torch.randn(3, 4)
    z = flow(x)
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-4)


def test_flow_backward() -> None:
    model = FlowEncoder(3)
    x = torch.randn(5, 3, requires_grad=True)
    out, tau = model(x)
    loss = (out + tau).mean()
    loss.backward()
    assert x.grad is not None
