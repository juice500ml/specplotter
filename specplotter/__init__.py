"""SpecPlotter - A tool for creating wideband spectrograms with signal analysis."""

from .specplotter import SpecPlotter

# Version is managed by python-semantic-release based on git tags
# The version in pyproject.toml is updated by semantic-release from git tags
# This reads from the installed package metadata (which comes from pyproject.toml)
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("specplotter")
except ImportError:
    # Fallback for development: read from pyproject.toml or use placeholder
    # In production, this should never be reached
    __version__ = "0.0.0"

__all__ = ["SpecPlotter"]
