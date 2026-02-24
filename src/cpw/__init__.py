"""Package initialisation for comp-photonics-workflows.

This module exposes the version and convenient aliases for core functions.
"""

from .solvers.rcwa_1d import rcwa_1d, convergence_scan, energy_check  # noqa: F401
from .workflows.sweep import run_sweep  # noqa: F401
from .workflows.inverse_fit import run_inverse_fit  # noqa: F401

__all__ = [
    "rcwa_1d",
    "convergence_scan",
    "energy_check",
    "run_sweep",
    "run_inverse_fit",
]

__version__ = "0.1.0"