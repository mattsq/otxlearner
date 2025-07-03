"""Trainer base class and implementations."""

from .base import BaseTrainer

__all__ = ["BaseTrainer", "SinkhornTrainer", "DANNTrainer"]


def __getattr__(name: str) -> type[BaseTrainer]:
    if name == "SinkhornTrainer":
        from .sinkhorn import SinkhornTrainer

        return SinkhornTrainer
    if name == "DANNTrainer":
        from .dann import DANNTrainer

        return DANNTrainer
    raise AttributeError(name)
