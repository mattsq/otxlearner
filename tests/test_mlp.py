from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import MLPEncoder
import torch


def test_mlp_encoder_shapes() -> None:
    batch_size = 4
    input_dim = 10
    model = MLPEncoder(input_dim)
    x = torch.randn(batch_size, input_dim)
    outcome, tau = model(x)
    assert outcome.shape == (batch_size,)
    assert tau.shape == (batch_size,)
