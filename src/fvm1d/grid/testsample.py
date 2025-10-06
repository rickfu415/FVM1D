from __future__ import annotations

from pathlib import Path

try:
    # When executed via package (e.g., `python -m fvm1d.grid.testsample`)
    from . import Grid1D  # type: ignore
    from ..io import write_tecplot_grid  # type: ignore
except Exception:
    # When executed directly in this folder (e.g., `python testsample.py`)
    from __init__ import Grid1D  # type: ignore
    # Add project src to path to import io writer
    import sys
    _this = Path(__file__).resolve()
    _repo_root = _this.parents[3]
    sys.path.append(str(_repo_root / "src"))
    from fvm1d.io import write_tecplot_grid  # type: ignore


def summarize_grid(length: float = 1.0, num_cells: int = 4) -> dict[str, int]:
    grid = Grid1D.from_length(length, num_cells)

    node_positions = grid.node_positions
    cells_to_nodes = grid.cells_to_nodes
    faces_to_nodes = grid.faces_to_nodes
    faces_to_cells = grid.faces_to_cells
    cells_to_faces = grid.cells_to_faces

    summary = {
        "num_cells": grid.num_cells,
        "num_nodes": int(node_positions.shape[0]),
        "num_faces": int(faces_to_nodes.shape[0]),
    }

    # Quick consistency checks (raise if inconsistent)
    assert cells_to_nodes.shape == (grid.num_cells, 8)
    assert faces_to_nodes.shape[1] == 4
    assert faces_to_cells.shape == (faces_to_nodes.shape[0], 2)
    assert cells_to_faces.shape == (grid.num_cells, 6)

    return summary


def main() -> None:
    length = 10.0
    num_cells = 10
    grid = Grid1D.from_length(length, num_cells)

    node_positions = grid.node_positions
    cells_to_nodes = grid.cells_to_nodes
    faces_to_nodes = grid.faces_to_nodes
    faces_to_cells = grid.faces_to_cells
    cells_to_faces = grid.cells_to_faces

    print("Grid summary:")
    print(f"  num_cells: {grid.num_cells}")
    print(f"  num_nodes: {node_positions.shape[0]}")
    print(f"  num_faces: {faces_to_nodes.shape[0]}")

    print("\nNodes (id: x y z):")
    for nid, (x, y, z) in enumerate(node_positions):
        print(f"  {nid}: {x:.6g} {y:.6g} {z:.6g}")

    areas = grid.face_areas

    print("\nFaces (id: nodes[4] | owner,neighbor | area):")
    for fid in range(faces_to_nodes.shape[0]):
        n0, n1, n2, n3 = faces_to_nodes[fid]
        owner, neigh = faces_to_cells[fid]
        print(f"  {fid}: [{n0}, {n1}, {n2}, {n3}] | {owner}, {neigh} | {areas[fid]:.6g}")

    print("\nCells (id: nodes[8] | faces[6]):")
    vols = grid.cell_volumes
    for cid in range(grid.num_cells):
        cn = cells_to_nodes[cid]
        cf = cells_to_faces[cid]
        cn_str = ", ".join(str(int(v)) for v in cn)
        cf_str = ", ".join(str(int(v)) for v in cf)
        print(f"  {cid}: [{cn_str}] | [{cf_str}] | vol={vols[cid]:.6g}")

    # Write Tecplot file to repository-root/output
    def _find_repo_root(start: Path) -> Path:
        p = start
        for candidate in [p] + list(p.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate
        # Fallback for known structure: src/fvm1d/grid/tests...
        return start.parents[3]

    repo_root = _find_repo_root(Path(__file__).resolve())
    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_tecplot_grid(grid, timestep=1, out_dir=out_dir)
    print(f"\nTecplot grid written: {out_path}")


if __name__ == "__main__":
    main()


