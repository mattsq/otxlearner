from __future__ import annotations

import torch
from torch import nn

from ..models import MLPEncoder, Sinkhorn, UnbalancedSinkhorn
from ..schedulers import cosine_warmup_lambda
from .base import BaseTrainer
from ..registries import register_trainer


@register_trainer("sinkhorn")
class SinkhornTrainer(BaseTrainer):
    """Trainer using Sinkhorn divergence for balancing."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_sizes: list[int] | None = None,
        lr: float = 1e-3,
        lambda_max: float = 1.0,
        epsilon: float = 0.05,
        unbalanced: bool = False,
        epochs: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        model = MLPEncoder(input_dim, hidden_sizes=hidden_sizes)
        super().__init__(model, device=device)
        self.div = (
            UnbalancedSinkhorn(blur=epsilon) if unbalanced else Sinkhorn(blur=epsilon)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_schedule = cosine_warmup_lambda(lambda_max, epochs)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], epoch: int, lam: float
    ) -> torch.Tensor:
        x, t, yf, mu0, mu1, e = batch
        x, t, yf, mu0, mu1, e = (
            x.to(self.device),
            t.to(self.device),
            yf.to(self.device),
            mu0.to(self.device),
            mu1.to(self.device),
            e.to(self.device),
        )
        assert isinstance(self.model, MLPEncoder)
        feats = self.model.net(x)
        outcome = self.model.outcome_head(feats).squeeze(-1)
        tau = self.model.tau_head(feats).squeeze(-1)
        y_pred = outcome + t * tau
        factual_loss = nn.functional.mse_loss(y_pred, yf)
        mu1_hat = outcome + tau
        d_hat = t * (yf - outcome) / e + (1 - t) * (mu1_hat - yf) / (1 - e)
        tau_loss = nn.functional.mse_loss(tau, d_hat)
        feats_t = feats[t.bool()]
        feats_c = feats[~t.bool()]
        if len(feats_t) > 0 and len(feats_c) > 0:
            bal_loss: torch.Tensor = self.div(feats_t, feats_c)
        else:
            bal_loss = torch.tensor(0.0, device=self.device)
        loss: torch.Tensor = factual_loss + tau_loss + lam * bal_loss
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], epoch: int
    ) -> tuple[float, float]:
        x, t, yf, mu0, mu1, e = batch
        x, t, yf, mu0, mu1, e = (
            x.to(self.device),
            t.to(self.device),
            yf.to(self.device),
            mu0.to(self.device),
            mu1.to(self.device),
            e.to(self.device),
        )
        assert isinstance(self.model, MLPEncoder)
        feats = self.model.net(x)
        tau = self.model.tau_head(feats).squeeze(-1)
        outcome = self.model.outcome_head(feats).squeeze(-1)
        mu1_hat = outcome + tau
        d_hat = t * (yf - outcome) / e + (1 - t) * (mu1_hat - yf) / (1 - e)
        mse = nn.functional.mse_loss(tau, d_hat, reduction="sum").item()
        feats_t = feats[t.bool()]
        feats_c = feats[~t.bool()]
        if len(feats_t) > 0 and len(feats_c) > 0:
            bal = self.div(feats_t, feats_c).item() * x.size(0)
        else:
            bal = 0.0
        return mse, bal


__all__ = ["SinkhornTrainer"]
