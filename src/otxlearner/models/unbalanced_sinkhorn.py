from __future__ import annotations

import importlib
import torch
from torch import nn
from typing import Any, Callable, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - hint-only imports
    from typing import Any as Array
else:  # pragma: no cover - runtime fallback
    Array = torch.Tensor

jax2torch: Optional[Callable[[Callable[..., Any]], Callable[..., Any]]]
try:  # pragma: no cover - optional deps
    _jax2torch_mod = importlib.import_module("jax2torch")
    jax2torch = cast(
        Callable[[Callable[..., Any]], Callable[..., Any]],
        getattr(_jax2torch_mod, "jax2torch"),
    )
    pointcloud = cast(Any, importlib.import_module("ott.geometry.pointcloud"))
    costs = cast(Any, importlib.import_module("ott.geometry.costs"))
    linear_problem = cast(
        Any, importlib.import_module("ott.problems.linear.linear_problem")
    )
    sinkhorn = cast(Any, importlib.import_module("ott.solvers.linear.sinkhorn"))
    _HAS_JAX = True
except Exception:  # pragma: no cover - optional deps
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
