"""fvm1d: A minimal 1D finite volume method framework."""

from .grid import Grid1D  # re-export from subpackage
from .system import ConservationLaw1D, LinearAdvection
from .solver import FiniteVolumeSolver1D
from . import io
from .variable import (
    FieldLocation,
    FieldData,
    ScalarField,
    MaterialProperty,
    Variables,
    # aliases
    FieldData1D,
    ScalarField1D,
    MaterialProperty1D,
    Variables1D,
)

__all__ = [
    "Grid1D",
    "ConservationLaw1D",
    "LinearAdvection",
    "FiniteVolumeSolver1D",
    "io",
    "FieldLocation",
    "FieldData",
    "ScalarField",
    "MaterialProperty",
    "Variables",
    # aliases
    "FieldData1D",
    "ScalarField1D",
    "MaterialProperty1D",
    "Variables1D",
]


