"""Evaluate a trained model on a dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import ModuleType
from typing import Optional, cast
import importlib

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import load_ihdp, load_twins, load_acic
from .data.types import DatasetProtocol
from .models import MLPEncoder, Sinkhorn
from .data.torch_adapter import torchify
from .utils import ate, pehe

_wandb: Optional[ModuleType]
try:  # pragma: no cover - optional dependency
    _wandb = importlib.import_module("wandb")
except Exception:
    _wandb = None
wandb: Optional[ModuleType] = _wandb


def evaluate(
    data_root: str | Path,
    model_path: str | Path,
    *,
    data: str = "ihdp",
    batch_size: int = 512,
    epsilon: float = 0.05,
    device: str | torch.device = "cpu",
    csv_path: str | Path | None = None,
    plot_file: str | Path | None = None,
    wandb_log: bool = False,
    wandb_project: str = "otxlearner",
) -> dict[str, float]:
    """Run evaluation on the specified dataset test split."""

    device = torch.device(device)
    if wandb_log and wandb is not None:
        wandb.init(project=wandb_project, job_type="evaluation")
    if data == "ihdp":
        ds_np = cast(DatasetProtocol, load_ihdp(data_root))
    elif data == "twins":
        ds_np = cast(DatasetProtocol, load_twins(data_root))
    else:
        ds_np = cast(DatasetProtocol, load_acic(data_root))
    ds = torchify(ds_np)
    loader = DataLoader(ds.test, batch_size=batch_size)

    model = MLPEncoder(ds.test.x.shape[1]).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    sinkhorn = Sinkhorn(blur=epsilon).to(device)

    model.eval()
    tau_err = 0.0
    bal = 0.0
    tau_preds: list[torch.Tensor] = []
    tau_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, t, _yf, mu0, mu1, _e in loader:
            x, t, mu0, mu1 = (
                x.to(device),
                t.to(device),
                mu0.to(device),
                mu1.to(device),
            )
            tau_true = mu1 - mu0
            feats = model.net(x)
            tau = model.tau_head(feats).squeeze(-1)
            tau_err += nn.functional.mse_loss(tau, tau_true, reduction="sum").item()
            feats_t = feats[t.bool()]
            feats_c = feats[~t.bool()]
            if len(feats_t) > 0 and len(feats_c) > 0:
                bal += sinkhorn(feats_t, feats_c).item() * x.size(0)
            tau_preds.append(tau.cpu())
            tau_targets.append(tau_true.cpu())

    mse = tau_err / len(ds.test)
    bal /= len(ds.test)
    tau_pred = torch.cat(tau_preds)
    tau_true = torch.cat(tau_targets)
    metrics = {
        "mse": mse,
        "balance": bal,
        "pehe": pehe(tau_pred, tau_true),
        "ate_error": ate(tau_pred, tau_true),
    }

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    if plot_file is not None:
        plt: Optional[ModuleType]
        try:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")
        except Exception:
            plt = None
        if plt is not None:
            tau_p = tau_pred.numpy()
            tau_t = tau_true.numpy()
            Path(plot_file).parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.scatter(tau_t, tau_p, s=5, alpha=0.5)
            plt.xlabel("True τ")
            plt.ylabel("Predicted τ")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()

    if wandb_log and wandb is not None:
        wandb.log(
            {
                "mse": mse,
                "balance": bal,
                "pehe": metrics["pehe"],
                "ate_error": metrics["ate_error"],
            }
        )
        wandb.finish()

    return metrics


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("model", type=Path, help="Path to .pt checkpoint")
    parser.add_argument("--data", choices=["ihdp", "twins", "acic"], default="ihdp")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B")
    args = parser.parse_args()

    root = args.data_root
    if root is None:
        root = Path.home() / f".cache/otxlearner/{args.data}"
    evaluate(
        root,
        args.model,
        data=args.data,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        csv_path=args.csv,
        plot_file=args.plot,
        wandb_log=args.wandb,
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
