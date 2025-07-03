#!/usr/bin/env python
"""Train Sinkhorn on IHDP quickly. Runtime <1 min."""
import argparse
import torch
from otxlearner.datasets import load_ihdp
from otxlearner.trainers import SinkhornTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    train, val, _ = load_ihdp(device=device)
    t = SinkhornTrainer(lambda_max=10, epsilon=0.1, device=device)
    t.fit(train, val, epochs=args.epochs)


if __name__ == "__main__":
    main()
