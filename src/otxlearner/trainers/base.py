from __future__ import annotations

import abc
from typing import Callable, Sized, cast

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseTrainer(abc.ABC):
    """Abstract base class for training loops."""

    def __init__(self, model: nn.Module, *, device: str | torch.device = "cpu"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer: torch.optim.Optimizer | None = None
        self.lambda_schedule: Callable[[int], float] | None = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    @abc.abstractmethod
    def training_step(
        self, batch: tuple[torch.Tensor, ...], epoch: int, lam: float
    ) -> torch.Tensor:
        """Compute the training loss for a batch."""

    @abc.abstractmethod
    def validation_step(
        self, batch: tuple[torch.Tensor, ...], epoch: int
    ) -> tuple[float, float]:
        """Return validation losses for a batch."""

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, ...]],
        val_loader: DataLoader[tuple[torch.Tensor, ...]],
        *,
        epochs: int,
        early_stop: int = 5,
    ) -> list[float]:
        """Run the training loop and return validation metric history."""

        if self.optimizer is None:
            raise RuntimeError("optimizer must be set")

        history: list[float] = []
        best_metric = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            lam = self.lambda_schedule(epoch) if self.lambda_schedule else 0.0
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.device.type):
                    loss = self.training_step(batch, epoch, lam)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += loss.item() * batch[0].size(0)

            _ = running_loss / len(cast(Sized, train_loader.dataset))

            # validation
            self.model.eval()
            val_mse = 0.0
            val_bal = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mse, bal = self.validation_step(batch, epoch)
                    val_mse += float(mse)
                    val_bal += float(bal)

            val_mse /= len(cast(Sized, val_loader.dataset))
            val_bal /= len(cast(Sized, val_loader.dataset))
            metric = val_mse + 0.1 * val_bal
            history.append(metric)
            if metric < best_metric:
                best_metric = metric
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop:
                    break

        return history


__all__ = ["BaseTrainer"]
