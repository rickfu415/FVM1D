from __future__ import annotations

from pathlib import Path

try:
    # When executed via package (e.g., `python -m fvm1d.variable.testsample`)
    from . import FieldData, Variables, MaterialProperty, Variable  # type: ignore
    from ..grid import Grid1D  # type: ignore
except Exception:
    # When executed directly in this folder (e.g., `python testsample.py`)
    import sys
    _this = Path(__file__).resolve()
    _repo_root = _this.parents[3]
    sys.path.append(str(_repo_root / "src"))
    from fvm1d.variable import FieldData, Variables, MaterialProperty, Variable  # type: ignore
    from fvm1d.grid import Grid1D  # type: ignore


def summarize_variables(length: float = 1.0, num_cells: int = 4) -> dict[str, int]:
    grid = Grid1D.from_length(length, num_cells)

    # Build a temperature field and material properties
    T = FieldData.from_function(grid, lambda x: 300.0 + 10.0 * x)
    rho = MaterialProperty.constant(grid, 1000.0)  # kg/m^3
    k = MaterialProperty.constant(grid, 0.6)       # W/(m·K)

    vars = Variables(grid)
    vars.set_T(T)
    vars.set_material(rho=rho, k=k)

    T_cells = vars.T.cell() if vars.T is not None else []
    T_faces = vars.T.face() if vars.T is not None else []
    T_nodes = vars.T.node() if vars.T is not None else []

    summary = {
        "num_cells": grid.num_cells,
        "len_T_cell": int(len(T_cells)),
        "len_T_face": int(len(T_faces)),
        "len_T_node": int(len(T_nodes)),
    }

    # Quick consistency checks (raise if inconsistent)
    assert summary["len_T_cell"] == grid.num_cells
    assert summary["len_T_face"] == grid.faces_to_nodes.shape[0]
    assert summary["len_T_node"] == grid.node_positions.shape[0]

    return summary


def main() -> None:
    length = 2.0
    num_cells = 10
    grid = Grid1D.from_length(length, num_cells)

    # Independent variables
    T = Variable.from_function(grid, "T", lambda x: 300.0 + 5.0 * x)
    Vx = Variable.from_constant(grid, "Vx", 0.0)

    rho = MaterialProperty.constant(grid, 998.2)
    k = MaterialProperty.constant(grid, 0.58)
    # Optional container remains usable alongside independent variables
    vars = Variables(grid)
    vars.set_T(T.data)
    vars.set_material(rho=rho, k=k)

    # Report
    print("Variables summary:")
    s = summarize_variables(length, num_cells)
    for kname, kval in s.items():
        print(f"  {kname}: {kval}")

    # Demonstrate property evaluation
    rho_field = rho.evaluate_field(T.data)
    print("\nProperty rho (constants) samples:")
    print(f"  cell[0]={rho_field.cell()[0]:.6g}, face[0]={rho_field.face()[0]:.6g}, node[0]={rho_field.node()[0]:.6g}")


if __name__ == "__main__":
    main()


