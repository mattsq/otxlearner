import torch
from torch.utils.data import DataLoader, TensorDataset

from otxlearner.trainers.base import BaseTrainer


class ToyTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__(torch.nn.Linear(1, 1))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.lambda_schedule = lambda epoch: 0.0

    def training_step(
        self, batch: tuple[torch.Tensor, ...], epoch: int, lam: float
    ) -> torch.Tensor:
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], epoch: int
    ) -> tuple[float, float]:
        x, y = batch
        pred = self.model(x)
        mse = torch.nn.functional.mse_loss(pred, y, reduction="sum")
        return float(mse), 0.0


def test_base_trainer_early_stop() -> None:
    data = torch.randn(10, 1)
    targets = torch.randn(10, 1)
    ds = TensorDataset(data, targets)
    loader = DataLoader(ds, batch_size=5)
    trainer = ToyTrainer()
    history = trainer.fit(loader, loader, epochs=5, early_stop=2)
    assert 1 <= len(history) <= 5
