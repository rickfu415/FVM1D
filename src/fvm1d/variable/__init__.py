from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

import numpy as np

from ..grid import Grid1D


class FieldLocation(str, Enum):
    CELL = "cell"
    FACE = "face"  # x-plane faces in 1D (N+1)
    NODE = "node"  # coincides with faces in 1D (N+1)


@dataclass(frozen=True)
class FieldData:
    """Field values located at cells, faces, and nodes on the grid.

    - ``cellData`` length: ``grid.num_cells``
    - ``faceData`` length: ``grid.faces_to_nodes.shape[0]``
    - ``nodeData`` length: ``grid.node_positions.shape[0]``

    Accessors ``cell()``, ``face()``, and ``node()`` return the respective arrays.
    """

    grid: Grid1D
    cellData: np.ndarray
    faceData: np.ndarray
    nodeData: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.cellData, dtype=float)
        f = np.asarray(self.faceData, dtype=float)
        n = np.asarray(self.nodeData, dtype=float)
        if c.ndim != 1 or c.shape[0] != self.grid.num_cells:
            raise ValueError("cellData must be 1D with length grid.num_cells")
        expected_faces = self.grid.faces_to_nodes.shape[0]
        expected_nodes = self.grid.node_positions.shape[0]
        if f.ndim != 1 or f.shape[0] != expected_faces:
            raise ValueError("faceData must be 1D with length grid.faces_to_nodes.shape[0]")
        if n.ndim != 1 or n.shape[0] != expected_nodes:
            raise ValueError("nodeData must be 1D with length grid.node_positions.shape[0]")
        object.__setattr__(self, "cellData", np.ascontiguousarray(c, dtype=float))
        object.__setattr__(self, "faceData", np.ascontiguousarray(f, dtype=float))
        object.__setattr__(self, "nodeData", np.ascontiguousarray(n, dtype=float))

    @classmethod
    def zeros(cls, grid: Grid1D) -> "FieldData":
        num_cells = grid.num_cells
        num_faces = grid.faces_to_nodes.shape[0]
        num_nodes = grid.node_positions.shape[0]
        return cls(
            grid=grid,
            cellData=np.zeros(num_cells, dtype=float),
            faceData=np.zeros(num_faces, dtype=float),
            nodeData=np.zeros(num_nodes, dtype=float),
        )

    @classmethod
    def constant(cls, grid: Grid1D, value: float) -> "FieldData":
        v = float(value)
        num_faces = grid.faces_to_nodes.shape[0]
        num_nodes = grid.node_positions.shape[0]
        return cls(
            grid=grid,
            cellData=np.full(grid.num_cells, v, dtype=float),
            faceData=np.full(num_faces, v, dtype=float),
            nodeData=np.full(num_nodes, v, dtype=float),
        )

    @classmethod
    def from_function(
        cls, grid: Grid1D, fn: Callable[[np.ndarray], np.ndarray | float]
    ) -> "FieldData":
        # Compute 3D coordinates for cell centers, face centers, and nodes
        node_xyz = grid.node_positions  # ((N+1)*4, 3)
        cell_nodes = grid.cells_to_nodes  # (N, 8)
        face_nodes = grid.faces_to_nodes  # (5N+1, 4)

        # Cell centers as mean of their 8 node coordinates
        cell_xyz = node_xyz[cell_nodes].mean(axis=1)
        # Face centers as mean of their 4 node coordinates
        face_xyz = node_xyz[face_nodes].mean(axis=1)

        def eval_fn(coords: np.ndarray) -> np.ndarray:
            # Try fn(coords) first; if not 1D per-point, fallback to x-only
            try:
                vals = fn(coords)
                arr = np.asarray(vals, dtype=float)
                if arr.ndim == 1 and arr.shape[0] == coords.shape[0]:
                    return arr
            except Exception:
                pass
            # Fallback: evaluate using x-coordinate only
            x_only = coords[:, 0]
            return np.asarray(fn(x_only), dtype=float)

        c = eval_fn(cell_xyz)
        f = eval_fn(face_xyz)
        n = eval_fn(node_xyz)
        return cls(grid=grid, cellData=c, faceData=f, nodeData=n)

    def copy(self) -> "FieldData":
        return FieldData(self.grid, self.cellData.copy(), self.faceData.copy(), self.nodeData.copy())

    def cell(self) -> np.ndarray:
        return self.cellData

    def face(self) -> np.ndarray:
        return self.faceData

    def node(self) -> np.ndarray:
        return self.nodeData


@dataclass(frozen=True)
class ScalarField:
    """Cell-centered scalar field on a 1D grid.

    Values are defined per control volume (cell) and stored as a 1D array of
    length ``grid.num_cells``.
    """

    grid: Grid1D
    values: np.ndarray
    location: FieldLocation = FieldLocation.CELL

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        expected = self._expected_length()
        if v.ndim != 1 or v.shape[0] != expected:
            raise ValueError(
                f"ScalarField1D length mismatch for {self.location}: expected {expected}, got {v.shape[0]}"
            )
        # Normalize to float64 contiguous array for downstream solvers/IO
        object.__setattr__(self, "values", np.ascontiguousarray(v, dtype=float))

    def _expected_length(self) -> int:
        if self.location == FieldLocation.CELL:
            return self.grid.num_cells
        if self.location == FieldLocation.FACE or self.location == FieldLocation.NODE:
            return self.grid.num_cells + 1
        raise ValueError("Unknown field location")

    @classmethod
    def constant(
        cls, grid: Grid1D, value: float, *, location: FieldLocation = FieldLocation.CELL
    ) -> "ScalarField":
        if location == FieldLocation.CELL:
            size = grid.num_cells
        elif location in (FieldLocation.FACE, FieldLocation.NODE):
            size = grid.num_cells + 1
        else:
            raise ValueError("Unknown field location")
        return cls(grid=grid, values=np.full(size, float(value)), location=location)

    @classmethod
    def from_function(
        cls,
        grid: Grid1D,
        fn: Callable[[np.ndarray], np.ndarray | float],
        *,
        location: FieldLocation = FieldLocation.CELL,
    ) -> "ScalarField":
        if location == FieldLocation.CELL:
            x = grid.cell_centers
        elif location in (FieldLocation.FACE, FieldLocation.NODE):
            x = grid.cell_edges
        else:
            raise ValueError("Unknown field location")
        vals = fn(x)
        return cls(grid=grid, values=np.asarray(vals, dtype=float), location=location)

    def copy(self) -> "ScalarField":
        return ScalarField(self.grid, self.values.copy(), self.location)


@dataclass(frozen=True)
class MaterialProperty:
    """Material property defined on a 1D grid.

    Supports constant properties now, and can be extended to temperature-dependent
    properties via a function ``temperature_fn(T_values)`` later.
    """

    grid: Grid1D
    constant_value: Optional[float] = None
    temperature_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        # No shape to validate here; evaluation will enforce shapes
        pass

    @classmethod
    def constant(
        cls, grid: Grid1D, value: float, *, location: FieldLocation = FieldLocation.CELL
    ) -> "MaterialProperty":
        # 'location' accepted for backward compatibility with evaluate()
        return cls(grid=grid, constant_value=float(value))

    def evaluate(self, T: Optional[ScalarField] = None) -> ScalarField:
        """Return the property as a ``ScalarField1D``.

        If ``temperature_fn`` is provided and ``T`` is given, compute
        ``temperature_fn(T.values)``; otherwise return the constant field.
        """
        if self.temperature_fn is not None:
            if T is None:
                raise ValueError("Temperature field T is required for temperature-dependent property")
            if T.grid is not self.grid:
                raise ValueError("Temperature field grid mismatch")
            values = np.asarray(self.temperature_fn(T.values), dtype=float)
            # Default to returning a cell-based field (historical behavior)
            return ScalarField(grid=self.grid, values=values, location=T.location)

        if self.constant_value is None:
            raise ValueError("MaterialProperty1D has neither constant_value nor temperature_fn")
        return ScalarField.constant(self.grid, self.constant_value, location=FieldLocation.CELL)

    def evaluate_field(self, T: Optional[FieldData] = None) -> FieldData:
        """Return the property as a ``FieldData`` with cell/face/node arrays.

        If a temperature function is provided, it is applied component-wise to
        the temperature arrays. Otherwise a constant field is returned.
        """
        if self.temperature_fn is not None:
            if T is None:
                raise ValueError("Temperature FieldData1D is required for temperature-dependent property")
            if T.grid is not self.grid:
                raise ValueError("Temperature FieldData1D grid mismatch")
            c = np.asarray(self.temperature_fn(T.cell()), dtype=float)
            f = np.asarray(self.temperature_fn(T.face()), dtype=float)
            n = np.asarray(self.temperature_fn(T.node()), dtype=float)
            return FieldData(self.grid, c, f, n)

        if self.constant_value is None:
            raise ValueError("MaterialProperty1D has neither constant_value nor temperature_fn")
        return FieldData.constant(self.grid, self.constant_value)


@dataclass
class Variables:
    """Container for solution variables and material properties on a 1D grid.

    Common fields (all optional, cell-centered):
    - Solution variables: T, P, vx, vy, vz
    - Material properties: rho (density), k (thermal conductivity), mu (viscosity), cp

    Arbitrary additional fields can be stored in the ``scalars`` and
    ``properties`` dictionaries.
    """

    grid: Grid1D

    # Common solution variables (optional), stored with cell/face/node arrays
    T: Optional[FieldData] = None
    P: Optional[FieldData] = None
    vx: Optional[FieldData] = None
    vy: Optional[FieldData] = None
    vz: Optional[FieldData] = None

    # Common material properties (optional)
    rho: Optional[MaterialProperty1D] = None  # density
    k: Optional[MaterialProperty1D] = None    # thermal conductivity
    mu: Optional[MaterialProperty1D] = None   # dynamic viscosity
    cp: Optional[MaterialProperty1D] = None   # specific heat capacity

    # Extensible maps
    scalars: Dict[str, FieldData] = field(default_factory=dict)
    properties: Dict[str, MaterialProperty1D] = field(default_factory=dict)

    def _validate_field(self, field: FieldData) -> None:
        if field.grid is not self.grid:
            raise ValueError("Field grid does not match Variables1D grid")

    # Convenience creators for constants
    def set_constant_property(
        self, name: str, value: float, *, location: FieldLocation = FieldLocation.CELL
    ) -> None:
        self.properties[name] = MaterialProperty1D.constant(self.grid, value, location=location)

    def set_constant_scalar(self, name: str, value: float) -> None:
        self.scalars[name] = FieldData.constant(self.grid, value)

    # Typed setters (validate and assign)
    def set_T(self, field: FieldData) -> None:
        self._validate_field(field)
        self.T = field

    def set_P(self, field: FieldData) -> None:
        self._validate_field(field)
        self.P = field

    def set_velocity(
        self,
        vx: Optional[FieldData] = None,
        vy: Optional[FieldData] = None,
        vz: Optional[FieldData] = None,
    ) -> None:
        for comp in (vx, vy, vz):
            if comp is not None:
                self._validate_field(comp)
        if vx is not None:
            self.vx = vx
        if vy is not None:
            self.vy = vy
        if vz is not None:
            self.vz = vz

    def set_material(
        self,
        *,
        rho: Optional[MaterialProperty1D] = None,
        k: Optional[MaterialProperty1D] = None,
        mu: Optional[MaterialProperty1D] = None,
        cp: Optional[MaterialProperty1D] = None,
    ) -> None:
        # Material properties are independent wrappers; validate grid match
        for prop in (rho, k, mu, cp):
            if prop is not None and prop.grid is not self.grid:
                raise ValueError("Material property grid does not match Variables1D grid")
        if rho is not None:
            self.rho = rho
        if k is not None:
            self.k = k
        if mu is not None:
            self.mu = mu
        if cp is not None:
            self.cp = cp

    # Extensible map helpers
    def add_scalar(self, name: str, field: FieldData) -> None:
        self._validate_field(field)
        self.scalars[name] = field

    def add_property(self, name: str, prop: MaterialProperty1D) -> None:
        if prop.grid is not self.grid:
            raise ValueError("Material property grid does not match Variables1D grid")
        self.properties[name] = prop

    def get(self, name: str):
        if name in self.scalars:
            return self.scalars[name]
        if name in self.properties:
            return self.properties[name]
        return getattr(self, name, None)


__all__ = [
    "FieldLocation",
    "FieldData",
    "ScalarField",
    "MaterialProperty",
    "Variables",
    "Variable",
]

# Backwards-compatible aliases
FieldData1D = FieldData
ScalarField1D = ScalarField
MaterialProperty1D = MaterialProperty
Variables1D = Variables


@dataclass
class Variable:
    """Independent variable bound to a grid with values at cells/faces/nodes.

    Examples: Temperature ("T"), Velocity-x ("Vx").
    """

    grid: Grid1D
    name: str
    data: FieldData

    def __post_init__(self) -> None:
        if self.data.grid is not self.grid:
            raise ValueError("Variable data grid mismatch")

    @classmethod
    def from_fielddata(cls, name: str, data: FieldData) -> "Variable":
        return cls(grid=data.grid, name=name, data=data)

    @classmethod
    def from_constant(cls, grid: Grid1D, name: str, value: float) -> "Variable":
        return cls(grid=grid, name=name, data=FieldData.constant(grid, value))

    @classmethod
    def from_function(
        cls, grid: Grid1D, name: str, fn: Callable[[np.ndarray], np.ndarray | float]
    ) -> "Variable":
        return cls(grid=grid, name=name, data=FieldData.from_function(grid, fn))

    def cell(self) -> np.ndarray:
        return self.data.cell()

    def face(self) -> np.ndarray:
        return self.data.face()

    def node(self) -> np.ndarray:
        return self.data.node()

    def set_data(self, data: FieldData) -> None:
        if data.grid is not self.grid:
            raise ValueError("Variable data grid mismatch")
        self.data = data


