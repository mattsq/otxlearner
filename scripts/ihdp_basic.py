#!/usr/bin/env python
"""Train Sinkhorn on IHDP quickly. Runtime <1 min."""
import argparse
import torch
from otxlearner.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    cli_main(
        [
            "ihdp",
            "sinkhorn",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
    )


if __name__ == "__main__":
    main()
