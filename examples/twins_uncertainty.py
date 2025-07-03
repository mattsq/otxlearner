#!/usr/bin/env python
"""Twins prediction intervals using MC-dropout. Runtime <2 min."""
import argparse
import torch
import numpy as np
from otxlearner.data import load_twins, torchify
from otxlearner.trainers import SinkhornTrainer
from otxlearner.utils import cross_fit_propensity
from otxlearner.loops import prepare_loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    ds_np = load_twins()
    x_all = np.concatenate([ds_np.train.x, ds_np.val.x, ds_np.test.x])
    t_all = np.concatenate([ds_np.train.t, ds_np.val.t, ds_np.test.t])
    e_all = cross_fit_propensity(x_all, t_all, n_splits=5, seed=0)
    n_tr = ds_np.train.x.shape[0]
    n_val = ds_np.val.x.shape[0]
    e_train = e_all[:n_tr]
    e_val = e_all[n_tr : n_tr + n_val]
    e_test = e_all[n_tr + n_val :]
    ds = torchify(ds_np, (e_train, e_val, e_test))
    for split in (ds.train, ds.val, ds.test):
        split.x = split.x.to(device)
        split.t = split.t.to(device)
        split.yf = split.yf.to(device)
        split.mu0 = split.mu0.to(device)
        split.mu1 = split.mu1.to(device)
        split.e = split.e.to(device)

    train_loader, val_loader = prepare_loaders(ds, batch_size=512, seed=0)
    trainer = SinkhornTrainer(
        ds.train.x.shape[1], lambda_max=10, epsilon=0.1, device=device, dropout=0.1
    )
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    preds = []
    for _ in range(args.mc_samples):
        out = trainer.model(ds.test.x)
        preds.append(out.detach().cpu().numpy())
    samples = np.stack(preds)
    lower = np.percentile(samples, 5, axis=0)
    upper = np.percentile(samples, 95, axis=0)
    widths = upper - lower
    print("mean interval width", widths.mean())


if __name__ == "__main__":
    main()
