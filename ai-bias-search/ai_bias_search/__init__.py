"""Top-level package for AI Bias Search project."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ai-bias-search")
except PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.1.0"

__all__ = ["__version__"]
