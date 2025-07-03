from __future__ import annotations

import torch
from torch import nn

from ..models import MLPEncoder, GradientReversal, DomainDiscriminator
from ..schedulers import cosine_warmup_lambda
from .base import BaseTrainer
from ..registries import register_trainer


@register_trainer("dann")
class DANNTrainer(BaseTrainer):
    """Domain-adversarial training using gradient reversal."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_sizes: list[int] | None = None,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        lambda_max: float = 1.0,
        epochs: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        model = MLPEncoder(input_dim, hidden_sizes=hidden_sizes)
        super().__init__(model, device=device)
        self.grl = GradientReversal()
        assert isinstance(self.model, MLPEncoder)
        feat_dim = self.model.tau_head.in_features
        self.discriminator = DomainDiscriminator(feat_dim, hidden_dim=hidden_dim)
        params = list(self.model.parameters()) + list(self.discriminator.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
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
        dom_pred = self.discriminator(self.grl(feats, lam))
        bal_loss = nn.functional.cross_entropy(dom_pred, t.long())
        loss: torch.Tensor = factual_loss + tau_loss + bal_loss
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
        dom_pred = self.discriminator(feats)
        bal = nn.functional.cross_entropy(dom_pred, t.long(), reduction="sum").item()
        return mse, bal


__all__ = ["DANNTrainer"]
