from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..grid import Grid1D
from ..system import ConservationLaw1D


@dataclass
class FiniteVolumeSolver1D:
    """First-order finite volume solver with Lax–Friedrichs numerical flux.

    Periodic boundary conditions are used by default. This is intended as a
    simple, dependable baseline that can be extended with higher-order
    reconstruction and other Riemann solvers.
    """

    grid: Grid1D
    law: ConservationLaw1D
    periodic: bool = True

    def _alpha(self, u_left: np.ndarray, u_right: np.ndarray) -> float:
        # Dissipation coefficient (global max for robustness)
        return float(self.law.max_characteristic_speed(np.concatenate([u_left, u_right])))

    def _lax_friedrichs_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        f_left = self.law.flux(u_left)
        f_right = self.law.flux(u_right)
        alpha = self._alpha(u_left, u_right)
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    def compute_stable_timestep(self, u: np.ndarray, cfl: float = 0.9) -> float:
        if not (0.0 < cfl <= 1.0):
            raise ValueError("CFL must be in (0, 1]")
        speed = float(self.law.max_characteristic_speed(u))
        if speed == 0.0:
            # No wave propagation; no restriction from CFL
            return np.inf
        return cfl * self.grid.dx / speed

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance one explicit Euler step using first-order FV with LF flux."""
        if u.shape[0] != self.grid.num_cells:
            raise ValueError("State vector length must match number of cells")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        num_cells = self.grid.num_cells
        # Build left/right states at interfaces with periodic wrap
        u_right = u.copy()
        u_left = np.roll(u, 1)

        # Compute numerical flux at interfaces i = 0..N (length N+1)
        # interface i uses (left cell i-1, right cell i)
        flux_interfaces = self._lax_friedrichs_flux(u_left, u_right)

        # For periodicity, also need flux at interface after the last cell
        # Shifted arrays already account for periodic neighbors
        flux_interfaces = np.concatenate((flux_interfaces, flux_interfaces[:1]))

        # Update cell averages: u^{n+1}_i = u^n_i - (dt/dx)(F_{i+1/2}-F_{i-1/2})
        flux_diff = flux_interfaces[1:] - flux_interfaces[:-1]
        u_new = u - (dt / self.grid.dx) * flux_diff
        return u_new


