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


def write_tecplot_grid(grid: Grid1D, timestep: int, out_dir: str | Path = ".") -> Path:
    """Write the grid geometry to a Tecplot ASCII file (FEBrick zone).

    The output is written as an unstructured hexahedral (FEBRICK) zone with
    variables X, Y, Z. File is named like ``timestep00001.plt`` inside
    ``out_dir``.

    Parameters
    ----------
    grid:
        The ``Grid1D`` instance to export.
    timestep:
        Integer timestep index used in the filename.
    out_dir:
        Directory to write the file into. Created if it does not exist.

    Returns
    -------
    Path
        The path to the written ``.plt`` file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"timestep{timestep:05d}.plt"
    file_path = out_path / filename

    # Geometry
    node_positions = grid.node_positions  # shape: ((N+1)*4, 3)
    cells_to_nodes = grid.cells_to_nodes  # shape: (N, 8)

    num_nodes = int(node_positions.shape[0])
    num_elems = int(cells_to_nodes.shape[0])

    # Write Tecplot ASCII
    with file_path.open("w", newline="\n") as f:
        f.write('TITLE = "Grid1D"\n')
        f.write('VARIABLES = "X" "Y" "Z"\n')
        f.write(
            f'ZONE T="grid", N={num_nodes}, E={num_elems}, ZONETYPE=FEBRICK, DATAPACKING=POINT\n'
        )

        # Node coordinates
        for x, y, z in node_positions:
            f.write(f"{x:.16e} {y:.16e} {z:.16e}\n")

        # Element connectivity (Tecplot uses 1-based indices)
        for conn in cells_to_nodes:
            idx = (conn + 1).astype(int)
            f.write(" ".join(str(i) for i in idx) + "\n")

    return file_path


