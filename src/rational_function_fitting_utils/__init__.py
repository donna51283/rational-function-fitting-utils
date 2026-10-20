"""Package entry point: re-export the public API."""

from .core import fit_rational, evaluate_rational, RationalFit

__all__ = ["fit_rational", "evaluate_rational", "RationalFit"]
