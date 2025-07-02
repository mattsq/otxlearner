from __future__ import annotations

import torch
from typing import Callable


"""Learning-rate and lambda schedulers."""


def cosine_warmup_lambda(
    max_lambda: float, num_epochs: int, warmup_frac: float = 0.1
) -> Callable[[int], float]:
    """Return λ schedule that ramps up during the first epochs."""

    warmup_epochs = max(1, int(num_epochs * warmup_frac))

    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            val = (
                max_lambda
                * (1 - torch.cos(torch.tensor(epoch / warmup_epochs * torch.pi)))
                / 2
            )
            return float(val.item())
        return float(max_lambda)

    return schedule


__all__ = ["cosine_warmup_lambda"]
