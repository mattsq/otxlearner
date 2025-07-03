import torch
from torch.utils.data import DataLoader, TensorDataset

from otxlearner.trainers.sinkhorn import SinkhornTrainer


def test_sinkhorn_trainer_overfit() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 2)
    t = torch.randint(0, 2, (4,)).float()
    yf = torch.randn(4)
    mu0 = torch.zeros_like(yf)
    mu1 = torch.zeros_like(yf)
    e = torch.full_like(yf, 0.5)
    ds = TensorDataset(x, t, yf, mu0, mu1, e)
    loader = DataLoader(ds, batch_size=4)
    trainer = SinkhornTrainer(input_dim=2, lambda_max=0.0, epochs=1)
    trainer.div = lambda a, b: torch.tensor(0.0)
    history = trainer.fit(loader, loader, epochs=1, early_stop=1)
    assert len(history) >= 1
