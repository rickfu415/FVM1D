from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Cells1D:
    """Collection of cell-centered data for a 1D uniform grid."""

    centers: np.ndarray
    widths: np.ndarray
    count: int


@dataclass(frozen=True)
class Faces1D:
    """Collection of face locations for a 1D uniform grid.

    In 1D, faces coincide with mesh nodes (vertices).
    """

    positions: np.ndarray
    count: int


@dataclass(frozen=True)
class Nodes1D:
    """Collection of node (vertex) locations for a 1D uniform grid.

    In 1D, nodes coincide with faces.
    """

    positions: np.ndarray
    count: int


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D finite volume grid.

    The grid stores cell-edge coordinates and derived cell-center coordinates.
    Periodicity is handled by the solver; the grid here is purely geometric.
    """

    x_start: float
    x_end: float
    num_cells: int
    y_bounds: tuple[float, float] = (0.0, 1.0)
    z_bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.x_start) or not np.isfinite(self.x_end):
            raise ValueError("Grid bounds must be finite numbers")
        if self.x_end <= self.x_start:
            raise ValueError("x_end must be greater than x_start")
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")
        y0, y1 = self.y_bounds
        z0, z1 = self.z_bounds
        if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(z0) and np.isfinite(z1)):
            raise ValueError("y/z bounds must be finite numbers")
        if y1 <= y0 or z1 <= z0:
            raise ValueError("Upper bounds must be greater than lower bounds for y/z")

    @classmethod
    def from_length(
        cls,
        length: float,
        num_cells: int,
        *,
        y_width: float | None = None,
        z_width: float | None = None,
    ) -> "Grid1D":
        """Construct a grid over [0, L] with given number of cells.

        By default, the widths in y and z are set to L/5, i.e.,
        y_bounds = (0, L/5) and z_bounds = (0, L/5).
        """
        L = float(length)
        wy = L / 5.0 if y_width is None else float(y_width)
        wz = L / 5.0 if z_width is None else float(z_width)
        return cls(0.0, L, num_cells, y_bounds=(0.0, wy), z_bounds=(0.0, wz))

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

    @property
    def faces(self) -> Faces1D:
        positions = self.cell_edges
        return Faces1D(positions=positions, count=positions.shape[0])

    @property
    def nodes(self) -> Nodes1D:
        positions = self.cell_edges
        return Nodes1D(positions=positions, count=positions.shape[0])

    @property
    def cells(self) -> Cells1D:
        centers = self.cell_centers
        widths = np.full(self.num_cells, self.dx)
        return Cells1D(centers=centers, widths=widths, count=self.num_cells)

    # Geometric measures
    @property
    def y_width(self) -> float:
        return float(self.y_bounds[1] - self.y_bounds[0])

    @property
    def z_width(self) -> float:
        return float(self.z_bounds[1] - self.z_bounds[0])

    @property
    def cell_volumes(self) -> np.ndarray:
        """Per-cell volumes for extruded hexahedra (constant for uniform mesh)."""
        volume = self.dx * self.y_width * self.z_width
        return np.full(self.num_cells, volume, dtype=float)

    @property
    def face_areas(self) -> np.ndarray:
        """Per-face areas ordered consistently with faces_to_nodes.

        - x-plane faces (N+1): area = y_width * z_width
        - y-faces (2N): area = dx * z_width
        - z-faces (2N): area = dx * y_width
        """
        N = self.num_cells
        areas = np.empty((N + 1) + 4 * N, dtype=float)
        ax = self.y_width * self.z_width
        ay = self.dx * self.z_width
        az = self.dx * self.y_width
        # x-planes
        areas[: N + 1] = ax
        # y faces (y0 then y1)
        areas[N + 1 : N + 1 + N] = ay
        areas[N + 1 + N : N + 1 + 2 * N] = ay
        # z faces (z0 then z1)
        areas[N + 1 + 2 * N : N + 1 + 3 * N] = az
        areas[N + 1 + 3 * N : N + 1 + 4 * N] = az
        return areas

    # ---------------------------- 3D topology (extruded along x) ----------------------------
    def _node_index(self, plane_index: int, corner_index: int) -> int:
        return 4 * plane_index + corner_index

    @property
    def node_positions(self) -> np.ndarray:
        """Return array of shape ((N+1)*4, 3) with (x,y,z) for each node.

        For each x-plane j in [0..N], the 4 nodes are ordered as:
            0: (y0, z0), 1: (y1, z0), 2: (y1, z1), 3: (y0, z1)
        """
        N = self.num_cells
        x_edges = self.cell_edges
        y0, y1 = self.y_bounds
        z0, z1 = self.z_bounds

        coords = np.empty(((N + 1) * 4, 3), dtype=float)
        corners_yz = np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]], dtype=float)
        idx = 0
        for j in range(N + 1):
            x = x_edges[j]
            for c in range(4):
                y, z = corners_yz[c]
                coords[idx, :] = (x, y, z)
                idx += 1
        return coords

    @property
    def cells_to_nodes(self) -> np.ndarray:
        """Return (N, 8) node indices for each hexahedral cell.

        Ordering: [left_plane c0..c3, right_plane c0..c3].
        """
        N = self.num_cells
        ctn = np.empty((N, 8), dtype=int)
        for i in range(N):
            left = [self._node_index(i, c) for c in range(4)]
            right = [self._node_index(i + 1, c) for c in range(4)]
            ctn[i, :] = left + right
        return ctn

    @property
    def faces_to_nodes(self) -> np.ndarray:
        """Return (5N+1, 4) node indices for each quad face.

        Face ordering:
        - First N+1 faces: x-constant planes at each edge j=0..N
        - Next N faces: y = y0 faces for each cell i
        - Next N faces: y = y1 faces for each cell i
        - Next N faces: z = z0 faces for each cell i
        - Next N faces: z = z1 faces for each cell i
        """
        N = self.num_cells
        total_faces = (N + 1) + 4 * N
        ftn = np.empty((total_faces, 4), dtype=int)

        # x-plane faces
        for j in range(N + 1):
            ftn[j, :] = [self._node_index(j, c) for c in range(4)]

        base = N + 1
        # y = y0 faces per cell
        for i in range(N):
            ftn[base + i, :] = [
                self._node_index(i, 0),
                self._node_index(i + 1, 0),
                self._node_index(i + 1, 3),
                self._node_index(i, 3),
            ]

        # y = y1 faces per cell
        base += N
        for i in range(N):
            ftn[base + i, :] = [
                self._node_index(i, 1),
                self._node_index(i + 1, 1),
                self._node_index(i + 1, 2),
                self._node_index(i, 2),
            ]

        # z = z0 faces per cell
        base += N
        for i in range(N):
            ftn[base + i, :] = [
                self._node_index(i, 0),
                self._node_index(i + 1, 0),
                self._node_index(i + 1, 1),
                self._node_index(i, 1),
            ]

        # z = z1 faces per cell
        base += N
        for i in range(N):
            ftn[base + i, :] = [
                self._node_index(i, 3),
                self._node_index(i + 1, 3),
                self._node_index(i + 1, 2),
                self._node_index(i, 2),
            ]

        return ftn

    @property
    def faces_to_cells(self) -> np.ndarray:
        """Return (5N+1, 2) mapping [owner, neighbor] cell indices; -1 for boundary."""
        N = self.num_cells
        total_faces = (N + 1) + 4 * N
        ftc = np.full((total_faces, 2), -1, dtype=int)

        # x-plane faces between cells (owner=j-1, neighbor=j)
        for j in range(N + 1):
            owner = j - 1
            neighbor = j if j < N else -1
            if owner >= 0:
                ftc[j, 0] = owner
            if neighbor >= 0:
                ftc[j, 1] = neighbor

        base = N + 1
        # y and z faces are boundary faces for each cell
        for k in range(4 * N):
            i = k % N
            ftc[base + k, 0] = i
            ftc[base + k, 1] = -1

        return ftc

    @property
    def cells_to_faces(self) -> np.ndarray:
        """Return (N, 6) mapping of each cell to its 6 face indices.

        Order per cell: [left(-x), right(+x), y0(-y), y1(+y), z0(-z), z1(+z)].
        """
        N = self.num_cells
        ctf = np.empty((N, 6), dtype=int)
        x_base = 0
        y0_base = (N + 1)
        y1_base = y0_base + N
        z0_base = y1_base + N
        z1_base = z0_base + N

        for i in range(N):
            ctf[i, 0] = x_base + i       # left (-x) at j=i
            ctf[i, 1] = x_base + i + 1   # right (+x) at j=i+1
            ctf[i, 2] = y0_base + i
            ctf[i, 3] = y1_base + i
            ctf[i, 4] = z0_base + i
            ctf[i, 5] = z1_base + i
        return ctf


