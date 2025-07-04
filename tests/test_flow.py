from __future__ import annotations

import torch
from otxlearner.models.flow import FlowEncoder, RealNVP


def test_flow_encoder_shapes() -> None:
    model = FlowEncoder(4)
    x = torch.randn(2, 4)
    out, tau = model(x)
    assert out.shape == (2,)
    assert tau.shape == (2,)


def test_flow_predict_tau() -> None:
    model = FlowEncoder(3)
    x = torch.randn(2, 3)
    _, tau = model(x)
    pred = model.predict_tau(x)
    assert torch.allclose(tau, pred)


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
