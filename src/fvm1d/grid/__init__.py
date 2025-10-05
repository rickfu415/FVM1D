from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D finite volume grid.

    The grid stores cell-edge coordinates and derived cell-center coordinates.
    Periodicity is handled by the solver; the grid here is purely geometric.
    """

    x_start: float
    x_end: float
    num_cells: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.x_start) or not np.isfinite(self.x_end):
            raise ValueError("Grid bounds must be finite numbers")
        if self.x_end <= self.x_start:
            raise ValueError("x_end must be greater than x_start")
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")

    @property
    def dx(self) -> float:
        return (self.x_end - self.x_start) / float(self.num_cells)

    @property
    def cell_edges(self) -> np.ndarray:
        # length = num_cells + 1
        return np.linspace(self.x_start, self.x_end, self.num_cells + 1)

    @property
    def cell_centers(self) -> np.ndarray:
        edges = self.cell_edges
        return 0.5 * (edges[:-1] + edges[1:])


