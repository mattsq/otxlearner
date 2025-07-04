from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from otxlearner.models import MLPEncoder


def test_one_epoch_loss_decreases() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    n = 256
    x = torch.randn(n, 10, generator=g)
    t = torch.randint(0, 2, (n,), generator=g).float()
    tau = torch.randn(n, generator=g)
    outcome = torch.randn(n, generator=g)
    yf = outcome + t * tau + 0.1 * torch.randn(n, generator=g)

    ds = TensorDataset(x, t, yf, tau)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=g)

    model = MLPEncoder(10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def eval_loss() -> float:
        loss = 0.0
        with torch.no_grad():
            for xb, tb, yb, taub in loader:
                out, tau_pred = model(xb)
                y_pred = out + tb * tau_pred
                factual = torch.nn.functional.mse_loss(y_pred, yb)
                tau_loss = torch.nn.functional.mse_loss(tau_pred, taub)
                loss += (factual + tau_loss).item()
        return loss

    initial = eval_loss()

    for xb, tb, yb, taub in loader:
        opt.zero_grad()
        out, tau_pred = model(xb)
        y_pred = out + tb * tau_pred
        factual = torch.nn.functional.mse_loss(y_pred, yb)
        tau_loss = torch.nn.functional.mse_loss(tau_pred, taub)
        loss = factual + tau_loss
        loss.backward()
        opt.step()

    final = eval_loss()
    assert final < initial
