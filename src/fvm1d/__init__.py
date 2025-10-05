"""fvm1d: A minimal 1D finite volume method framework."""

from .grid import Grid1D  # re-export from subpackage
from .system import ConservationLaw1D, LinearAdvection
from .solver import FiniteVolumeSolver1D
from . import io

__all__ = [
    "Grid1D",
    "ConservationLaw1D",
    "LinearAdvection",
    "FiniteVolumeSolver1D",
    "io",
]


