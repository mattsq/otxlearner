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
