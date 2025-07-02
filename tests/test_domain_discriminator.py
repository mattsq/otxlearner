from __future__ import annotations

import torch

from otxlearner.models.domain import DomainDiscriminator


def test_domain_discriminator_output() -> None:
    model = DomainDiscriminator(5, hidden_dim=8)
    x = torch.randn(3, 5)
    out = model(x)
    assert out.shape == (3, 2)
