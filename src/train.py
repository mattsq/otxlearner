"""Minimal training loop for the IHDP dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import numpy.typing as npt

from .data import IHDPDataset, IHDPSplit, load_ihdp
from .models import MLPEncoder, Sinkhorn


@dataclass
class TorchSplit(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """Torch dataset wrapper around :class:`IHDPSplit`."""

    x: torch.Tensor
    t: torch.Tensor
    yf: torch.Tensor
    mu0: torch.Tensor
    mu1: torch.Tensor

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.x[idx],
            self.t[idx],
            self.yf[idx],
            self.mu0[idx],
            self.mu1[idx],
        )


@dataclass
class TorchIHDP:
    train: TorchSplit
    val: TorchSplit
    test: TorchSplit


def _to_tensor(x: torch.Tensor | npt.NDArray[np.floating]) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def torchify(ds: IHDPDataset) -> TorchIHDP:
    """Convert numpy IHDP dataset to PyTorch tensors."""

    def convert(split: IHDPSplit) -> TorchSplit:
        return TorchSplit(
            _to_tensor(split.x),
            _to_tensor(split.t),
            _to_tensor(split.yf),
            _to_tensor(split.mu0),
            _to_tensor(split.mu1),
        )

    return TorchIHDP(convert(ds.train), convert(ds.val), convert(ds.test))


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


def train(
    root: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_max: float = 1.0,
    epsilon: float = 0.05,
    patience: int = 5,
    device: str | torch.device = "cpu",
    log_dir: str | Path | None = None,
) -> None:
    """Train the baseline model on IHDP."""

    device = torch.device(device)
    ds_np = load_ihdp(root)
    ds = torchify(ds_np)

    train_loader = DataLoader(ds.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds.val, batch_size=batch_size)

    model = MLPEncoder(ds.train.x.shape[1]).to(device)
    sinkhorn = Sinkhorn(blur=epsilon).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    schedule = cosine_warmup_lambda(lambda_max, epochs)

    writer = SummaryWriter(log_dir=str(log_dir) if log_dir else None)  # type: ignore[no-untyped-call]

    best_metric = float("inf")
    epochs_without_improve = 0

    for epoch in range(epochs):
        lam = schedule(epoch)
        writer.add_scalar("lambda", lam, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("epsilon", epsilon, epoch)  # type: ignore[no-untyped-call]
        model.train()
        running_loss = 0.0
        for x, t, yf, mu0, mu1 in train_loader:
            x, t, yf, mu0, mu1 = (
                x.to(device),
                t.to(device),
                yf.to(device),
                mu0.to(device),
                mu1.to(device),
            )
            tau_true = mu1 - mu0
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feats = model.net(x)
                outcome = model.outcome_head(feats).squeeze(-1)
                tau = model.tau_head(feats).squeeze(-1)
                y_pred = outcome + t * tau
                factual_loss = nn.functional.mse_loss(y_pred, yf)
                tau_loss = nn.functional.mse_loss(tau, tau_true)
                feats_t = feats[t.bool()]
                feats_c = feats[~t.bool()]
                if len(feats_t) > 0 and len(feats_c) > 0:
                    bal_loss = sinkhorn(feats_t, feats_c)
                else:
                    bal_loss = torch.tensor(0.0, device=device)
                loss = factual_loss + tau_loss + lam * bal_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(ds.train)
        writer.add_scalar("train_loss", avg_loss, epoch)  # type: ignore[no-untyped-call]

        # validation
        model.eval()
        val_tau_err = 0.0
        val_bal = 0.0
        tau_preds: list[torch.Tensor] = []
        tau_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for x, t, yf, mu0, mu1 in val_loader:
                x, t, mu0, mu1 = (
                    x.to(device),
                    t.to(device),
                    mu0.to(device),
                    mu1.to(device),
                )
                tau_true = mu1 - mu0
                feats = model.net(x)
                tau = model.tau_head(feats).squeeze(-1)
                val_tau_err += nn.functional.mse_loss(
                    tau, tau_true, reduction="sum"
                ).item()
                feats_t = feats[t.bool()]
                feats_c = feats[~t.bool()]
                if len(feats_t) > 0 and len(feats_c) > 0:
                    val_bal += sinkhorn(feats_t, feats_c).item() * x.size(0)
                tau_preds.append(tau)
                tau_targets.append(tau_true)

        val_mse = val_tau_err / len(ds.val)
        val_bal /= len(ds.val)
        val_metric = val_mse + 0.1 * val_bal
        writer.add_scalar("val_PEHE_proxy", val_metric, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("val_mse", val_mse, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("val_bal", val_bal, epoch)  # type: ignore[no-untyped-call]

        pehe = torch.sqrt(torch.tensor(val_mse)).item()
        ate_pred = torch.cat(tau_preds).mean().item()
        ate_true = torch.cat(tau_targets).mean().item()
        ate_err = ate_pred - ate_true
        writer.add_scalar("val_PEHE", pehe, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("val_ATE_error", ate_err, epoch)  # type: ignore[no-untyped-call]

        if val_metric < best_metric:
            best_metric = val_metric
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    writer.close()  # type: ignore[no-untyped-call]


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Train baseline on IHDP")
    parser.add_argument(
        "--data-root", type=Path, default=Path.home() / ".cache/otxlearner/ihdp"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    train(
        args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_max=args.lambda_max,
        epsilon=args.epsilon,
        patience=args.patience,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
