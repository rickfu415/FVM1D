from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ConservationLaw1D(ABC):
    """Abstract base class for 1D conservation laws.

    Defines the flux function and a characteristic speed bound used by solvers.
    Implementations should vectorize over numpy arrays where reasonable.
    """

    @abstractmethod
    def flux(self, u: np.ndarray) -> np.ndarray:
        """Return flux f(u)."""

    @abstractmethod
    def max_characteristic_speed(self, u: np.ndarray) -> float:
        """Return max |lambda(u)| over provided states u.

        Used to set dissipation in e.g. Lax–Friedrichs flux and CFL timestep.
        """


class LinearAdvection(ConservationLaw1D):
    """u_t + a u_x = 0 with constant advection speed a."""

    def __init__(self, speed: float) -> None:
        self.speed = float(speed)

    def flux(self, u: np.ndarray) -> np.ndarray:
        return self.speed * u

    def max_characteristic_speed(self, u: np.ndarray) -> float:
        return abs(self.speed)


