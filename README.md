# fvm1d

A minimal, extensible 1D finite volume method (FVM) framework in Python.

## Features

- Uniform 1D grids (`Grid1D`)
- Conservation law interface (`ConservationLaw1D`) with example `LinearAdvection`
- Simple first-order Lax–Friedrichs finite volume solver (`FiniteVolumeSolver1D`)
- Lightweight CSV output utilities

## Install (editable)

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
from fvm1d import Grid1D, LinearAdvection, FiniteVolumeSolver1D

grid = Grid1D(x_start=0.0, x_end=1.0, num_cells=200)
law = LinearAdvection(speed=1.0)
solver = FiniteVolumeSolver1D(grid=grid, law=law)

x = grid.cell_centers
u = np.exp(-200.0 * (x - 0.5) ** 2)

dt = solver.compute_stable_timestep(u, cfl=0.9)
u_next = solver.step(u, dt)
```

## License

MIT


