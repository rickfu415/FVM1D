import numpy as np

from fvm1d.grid import Grid1D


def test_from_length_and_entities():
    L = 2.0
    N = 4
    g = Grid1D.from_length(L, N)

    assert g.x_start == 0.0
    assert g.x_end == L
    assert g.num_cells == N
    assert np.isclose(g.dx, L / N)

    edges = g.cell_edges
    assert edges.shape == (N + 1,)
    assert np.isclose(edges[0], 0.0)
    assert np.isclose(edges[-1], L)

    centers = g.cell_centers
    assert centers.shape == (N,)

    # Entities
    cells = g.cells
    faces = g.faces
    nodes = g.nodes

    assert cells.count == N
    assert faces.count == N + 1
    assert nodes.count == N + 1
    assert np.allclose(cells.widths, g.dx)
    assert np.allclose(faces.positions, edges)
    assert np.allclose(nodes.positions, edges)


def test_3d_topology_counts_and_connectivity():
    L = 1.0
    N = 3
    g = Grid1D.from_length(L, N)

    # Nodes: (N+1) planes * 4 per plane
    node_pos = g.node_positions
    assert node_pos.shape == ((N + 1) * 4, 3)

    # Cells->nodes: N hexahedra * 8 nodes each
    ctn = g.cells_to_nodes
    assert ctn.shape == (N, 8)

    # Faces: (N+1) x-planes + 4N side faces
    ftn = g.faces_to_nodes
    assert ftn.shape == ((N + 1) + 4 * N, 4)

    ftc = g.faces_to_cells
    assert ftc.shape == (ftn.shape[0], 2)

    ctf = g.cells_to_faces
    assert ctf.shape == (N, 6)

    # Check a known mapping for cell 0
    # left face is x-plane j=0, right face j=1
    assert ctf[0, 0] == 0
    assert ctf[0, 1] == 1


def test_measures_defaults_and_shapes():
    L = 5.0
    N = 4
    g = Grid1D.from_length(L, N)

    # Default y/z widths = L/5 = 1.0
    assert np.isclose(g.y_width, 1.0)
    assert np.isclose(g.z_width, 1.0)

    vols = g.cell_volumes
    assert vols.shape == (N,)
    assert np.allclose(vols, g.dx * g.y_width * g.z_width)

    areas = g.face_areas
    assert areas.shape == ((N + 1) + 4 * N,)
    # First N+1 x-faces
    assert np.allclose(areas[: N + 1], g.y_width * g.z_width)
    # Next 2N y-faces
    assert np.allclose(areas[N + 1 : N + 1 + 2 * N], g.dx * g.z_width)
    # Last 2N z-faces
    assert np.allclose(areas[N + 1 + 2 * N :], g.dx * g.y_width)


