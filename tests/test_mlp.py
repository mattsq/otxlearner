from __future__ import annotations

from otxlearner import MLPEncoder
import torch


def test_mlp_encoder_shapes() -> None:
    batch_size = 4
    input_dim = 10
    model = MLPEncoder(input_dim)
    x = torch.randn(batch_size, input_dim)
    outcome, tau = model(x)
    assert outcome.shape == (batch_size,)
    assert tau.shape == (batch_size,)


def test_mlp_predict_tau() -> None:
    model = MLPEncoder(3)
    x = torch.randn(2, 3)
    direct = model(x)[1]
    pred = model.predict_tau(x)
    assert torch.allclose(direct, pred)
