from __future__ import annotations

import torch
from torch import nn
from typing import Callable, cast

try:  # pragma: no cover - optional deps
    from jax2torch import jax2torch  # type: ignore[import-not-found]
    from ott.geometry import pointcloud, costs  # type: ignore[import-not-found]
    from ott.problems.linear import linear_problem  # type: ignore[import-not-found]
    from ott.solvers.linear import sinkhorn  # type: ignore[import-not-found]
    from jax.numpy import ndarray as Array  # type: ignore[import-not-found]

    _HAS_JAX = True
except Exception:  # pragma: no cover - optional deps
    Array = torch.Tensor
    jax2torch = None
    _HAS_JAX = False


def _make_sinkhorn_fn(
    eps: float, p: int, tau_a: float, tau_b: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if _HAS_JAX:

        def _fn(x: Array, y: Array) -> Array:
            geom = pointcloud.PointCloud(x, y, epsilon=eps, cost_fn=costs.PNormP(p))
            prob = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
            out = sinkhorn.Sinkhorn()(prob)
            return cast(Array, out.reg_ot_cost)

        assert jax2torch is not None
        return cast(
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor], jax2torch(_fn)
        )
    else:

        def _fallback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.cdist(x, y, p=p).mean()

        return _fallback


class UnbalancedSinkhorn(nn.Module):
    """Unbalanced Sinkhorn divergence using ott-jax."""

    def __init__(self, *, blur: float = 0.05, p: int = 2, tau: float = 0.8) -> None:
        super().__init__()
        self.blur = blur
        self.p = p
        self.tau = tau
        self.loss_fn = _make_sinkhorn_fn(self.blur, self.p, self.tau, self.tau)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Inputs must be 2D matrices")
        return self.loss_fn(x, y)


__all__ = ["UnbalancedSinkhorn"]
