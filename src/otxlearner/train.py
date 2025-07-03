"""Compatibility wrapper for the training CLI."""

from .cli import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
