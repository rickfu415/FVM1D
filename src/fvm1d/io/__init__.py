from __future__ import annotations

from pathlib import Path
import numpy as np

from ..grid import Grid1D


def write_csv(path: str | Path, grid: Grid1D, u: np.ndarray) -> None:
    """Write two-column CSV (x, u) for cell centers and values."""
    path = Path(path)
    data = np.column_stack([grid.cell_centers, u])
    header = "x,u"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


