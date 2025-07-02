from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Type

from .trainers.base import BaseTrainer

_DATASETS: Dict[str, Callable[..., object]] = {}
_TRAINERS: Dict[str, Type[BaseTrainer]] = {}


def register_dataset(
    name: str,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Register a dataset loader by name."""

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        _DATASETS[name] = fn
        return fn

    return decorator


def register_trainer(name: str) -> Callable[[Type[BaseTrainer]], Type[BaseTrainer]]:
    """Register a trainer class by name."""

    def decorator(cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
        _TRAINERS[name] = cls
        return cls

    return decorator


__all__ = ["_DATASETS", "_TRAINERS", "register_dataset", "register_trainer"]
