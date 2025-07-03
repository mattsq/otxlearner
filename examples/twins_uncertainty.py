#!/usr/bin/env python
"""Twins prediction intervals using MC-dropout. Runtime <2 min."""
import argparse
import torch
import numpy as np
from otxlearner.datasets import load_twins
from otxlearner.trainers import SinkhornTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    train, val, test = load_twins(device=device)
    trainer = SinkhornTrainer(lambda_max=10, epsilon=0.1, device=device, dropout=0.1)
    trainer.fit(train, val, epochs=args.epochs)
    preds = []
    for _ in range(args.mc_samples):
        out = trainer.model(test.x)
        preds.append(out.detach().cpu().numpy())
    samples = np.stack(preds)
    lower = np.percentile(samples, 5, axis=0)
    upper = np.percentile(samples, 95, axis=0)
    widths = upper - lower
    print("mean interval width", widths.mean())


if __name__ == "__main__":
    main()
