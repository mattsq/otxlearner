"""Minimal training loop for the IHDP dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from types import ModuleType
import importlib

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from .data import IHDPDataset, IHDPSplit, load_ihdp
from .models import MLPEncoder, Sinkhorn, FlowEncoder
from .utils import cross_fit_propensity

_wandb: Optional[ModuleType]
try:  # pragma: no cover - optional dependency
    _wandb = importlib.import_module("wandb")
except Exception:
    _wandb = None
wandb: Optional[ModuleType] = _wandb


@dataclass
class TorchSplit(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """Torch dataset wrapper around :class:`IHDPSplit`."""

    x: torch.Tensor
    t: torch.Tensor
    yf: torch.Tensor
    mu0: torch.Tensor
    mu1: torch.Tensor
    e: torch.Tensor

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.x[idx],
            self.t[idx],
            self.yf[idx],
            self.mu0[idx],
            self.mu1[idx],
            self.e[idx],
        )


@dataclass
class TorchIHDP:
    train: TorchSplit
    val: TorchSplit
    test: TorchSplit


@dataclass
class TrainConfig:
    """Configuration for :func:`train`."""

    data_root: Path = Path.home() / ".cache/otxlearner/ihdp"
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    lambda_max: float = 1.0
    epsilon: float = 0.05
    patience: int = 5
    device: str = "cpu"
    log_dir: Path | None = None
    seed: int = 42
    wandb: bool = False
    depth: int = 3
    width: int = 64
    encoder: str = "mlp"


cs = ConfigStore.instance()
cs.store(name="base", node=TrainConfig)


def _to_tensor(x: torch.Tensor | npt.NDArray[np.float64]) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def torchify(
    ds: IHDPDataset,
    propensities: (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | None
    ) = None,
) -> TorchIHDP:
    """Convert numpy IHDP dataset to PyTorch tensors."""

    def convert(split: IHDPSplit, e: npt.NDArray[np.float64]) -> TorchSplit:
        return TorchSplit(
            _to_tensor(split.x),
            _to_tensor(split.t),
            _to_tensor(split.yf),
            _to_tensor(split.mu0),
            _to_tensor(split.mu1),
            _to_tensor(e),
        )

    if propensities is None:
        zeros_train = np.zeros(len(ds.train.x), dtype=np.float64)
        zeros_val = np.zeros(len(ds.val.x), dtype=np.float64)
        zeros_test = np.zeros(len(ds.test.x), dtype=np.float64)
        propensities = (zeros_train, zeros_val, zeros_test)

    e_train, e_val, e_test = propensities
    return TorchIHDP(
        convert(ds.train, e_train),
        convert(ds.val, e_val),
        convert(ds.test, e_test),
    )


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
    depth: int = 3,
    width: int = 64,
    encoder: str = "mlp",
    device: str | torch.device = "cpu",
    log_dir: str | Path | None = None,
    seed: int = 42,
    wandb_log: bool = False,
    wandb_project: str = "otxlearner",
) -> list[float]:
    """Train the baseline model on IHDP and return validation metrics."""

    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_np = load_ihdp(root)
    x_all = np.concatenate([ds_np.train.x, ds_np.val.x, ds_np.test.x])
    t_all = np.concatenate([ds_np.train.t, ds_np.val.t, ds_np.test.t])
    e_all = cross_fit_propensity(x_all, t_all, n_splits=5, seed=seed)
    n_tr = ds_np.train.x.shape[0]
    n_val = ds_np.val.x.shape[0]
    e_train = e_all[:n_tr]
    e_val = e_all[n_tr : n_tr + n_val]
    e_test = e_all[n_tr + n_val :]
    ds = torchify(ds_np, (e_train, e_val, e_test))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        ds.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(ds.val, batch_size=batch_size)

    hidden = None if (width == 64 and depth == 3) else [width] * depth
    model: nn.Module
    if encoder == "flow":
        model = FlowEncoder(ds.train.x.shape[1], n_flows=depth, hidden_dim=width).to(
            device
        )
    else:
        model = MLPEncoder(ds.train.x.shape[1], hidden_sizes=hidden).to(device)
    sinkhorn = Sinkhorn(blur=epsilon).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    schedule = cosine_warmup_lambda(lambda_max, epochs)

    writer = SummaryWriter(log_dir=str(log_dir) if log_dir else None)  # type: ignore[no-untyped-call]
    if wandb_log and wandb is not None:
        wandb.init(
            project=wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "lambda_max": lambda_max,
                "epsilon": epsilon,
                "patience": patience,
                "seed": seed,
                "depth": depth,
                "width": width,
                "encoder": encoder,
            },
        )

    best_metric = float("inf")
    epochs_without_improve = 0
    val_history: list[float] = []

    for epoch in range(epochs):
        lam = schedule(epoch)
        writer.add_scalar("lambda", lam, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("epsilon", epsilon, epoch)  # type: ignore[no-untyped-call]
        if wandb_log and wandb is not None:
            wandb.log({"lambda": lam, "epsilon": epsilon}, step=epoch)
        model.train()
        running_loss = 0.0
        for x, t, yf, mu0, mu1, e in train_loader:
            x, t, yf, mu0, mu1, e = (
                x.to(device),
                t.to(device),
                yf.to(device),
                mu0.to(device),
                mu1.to(device),
                e.to(device),
            )
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                if isinstance(model, FlowEncoder):
                    feats = model.flow(x)
                    outcome = model.outcome_head(feats).squeeze(-1)
                    tau = model.tau_head(feats).squeeze(-1)
                else:
                    assert isinstance(model, MLPEncoder)
                    feats = model.net(x)
                    outcome = model.outcome_head(feats).squeeze(-1)
                    tau = model.tau_head(feats).squeeze(-1)
                y_pred = outcome + t * tau
                factual_loss = nn.functional.mse_loss(y_pred, yf)
                mu1_hat = outcome + tau
                d_hat = t * (yf - outcome) / e + (1 - t) * (mu1_hat - yf) / (1 - e)
                tau_loss = nn.functional.mse_loss(tau, d_hat)
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
        if wandb_log and wandb is not None:
            wandb.log({"train_loss": avg_loss}, step=epoch)

        # validation
        model.eval()
        val_tau_err = 0.0
        val_bal = 0.0
        tau_preds: list[torch.Tensor] = []
        tau_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for x, t, yf, mu0, mu1, e in val_loader:
                x, t, yf, mu0, mu1, e = (
                    x.to(device),
                    t.to(device),
                    yf.to(device),
                    mu0.to(device),
                    mu1.to(device),
                    e.to(device),
                )
                tau_true = mu1 - mu0
                if isinstance(model, FlowEncoder):
                    feats = model.flow(x)
                    tau = model.tau_head(feats).squeeze(-1)
                    outcome = model.outcome_head(feats).squeeze(-1)
                else:
                    assert isinstance(model, MLPEncoder)
                    feats = model.net(x)
                    tau = model.tau_head(feats).squeeze(-1)
                    outcome = model.outcome_head(feats).squeeze(-1)
                mu1_hat = outcome + tau
                d_hat = t * (yf - outcome) / e + (1 - t) * (mu1_hat - yf) / (1 - e)
                val_tau_err += nn.functional.mse_loss(
                    tau, d_hat, reduction="sum"
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
        val_history.append(val_metric)
        writer.add_scalar("val_PEHE_proxy", val_metric, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("val_mse", val_mse, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("val_bal", val_bal, epoch)  # type: ignore[no-untyped-call]
        if wandb_log and wandb is not None:
            wandb.log(
                {"val_PEHE_proxy": val_metric, "val_mse": val_mse, "val_bal": val_bal},
                step=epoch,
            )

        pehe = torch.sqrt(torch.tensor(val_mse)).item()
        ate_pred = torch.cat(tau_preds).mean().item()
        ate_true = torch.cat(tau_targets).mean().item()
        ate_err = ate_pred - ate_true
        writer.add_scalar("val_PEHE", pehe, epoch)  # type: ignore[no-untyped-call]
        writer.add_scalar("val_ATE_error", ate_err, epoch)  # type: ignore[no-untyped-call]
        if wandb_log and wandb is not None:
            wandb.log({"val_PEHE": pehe, "val_ATE": ate_pred}, step=epoch)

        if val_metric < best_metric:
            best_metric = val_metric
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    writer.close()  # type: ignore[no-untyped-call]
    if wandb_log and wandb is not None:
        wandb.finish()

    return val_history


def train_from_config(cfg: TrainConfig) -> list[float]:
    """Train using a :class:`TrainConfig`."""

    return train(
        cfg.data_root,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        lambda_max=cfg.lambda_max,
        epsilon=cfg.epsilon,
        patience=cfg.patience,
        depth=cfg.depth,
        width=cfg.width,
        encoder=cfg.encoder,
        device=cfg.device,
        log_dir=cfg.log_dir,
        seed=cfg.seed,
        wandb_log=cfg.wandb,
    )


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Train baseline on IHDP")
    parser.add_argument("--config", type=Path, default=None, help="Hydra config file")
    parser.add_argument(
        "--data-root", type=Path, default=Path.home() / ".cache/otxlearner/ihdp"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "flow"])
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B")

    args, unknown = parser.parse_known_args()

    if args.config is not None:
        config_path = args.config.resolve()
        # remove --config from sys.argv so Hydra can parse remaining overrides
        import sys

        idx = sys.argv.index("--config")
        del sys.argv[idx : idx + 2]
        with initialize(config_path=str(config_path.parent), version_base=None):
            cfg = compose(config_name=config_path.stem, overrides=unknown)
        train_from_config(cfg)  # type: ignore[arg-type]
    else:
        train(
            args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_max=args.lambda_max,
            epsilon=args.epsilon,
            patience=args.patience,
            depth=args.depth,
            width=args.width,
            encoder=args.encoder,
            log_dir=args.log_dir,
            wandb_log=args.wandb,
        )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
