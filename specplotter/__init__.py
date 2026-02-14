"""SpecPlotter - A tool for creating wideband spectrograms with signal analysis."""

from .specplotter import SpecPlotter

# Version is read from git tags via setuptools-scm at build time
# Git tags are the single source of truth
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["SpecPlotter"]
