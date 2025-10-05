import numpy as np

from fvm1d import Grid1D, LinearAdvection, FiniteVolumeSolver1D


def test_one_step_shapes_and_values():
    grid = Grid1D(0.0, 1.0, 128)
    law = LinearAdvection(speed=1.0)
    solver = FiniteVolumeSolver1D(grid=grid, law=law)

    x = grid.cell_centers
    u0 = np.exp(-200.0 * (x - 0.5) ** 2)

    dt = solver.compute_stable_timestep(u0, cfl=0.9)
    assert dt > 0

    u1 = solver.step(u0, dt)
    assert u1.shape == u0.shape
    assert np.all(np.isfinite(u1))


