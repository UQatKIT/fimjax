# FIMJAX

This Python project is an implementation of an eikonal solver on meshes.


## Features

- JAX-based implementation for GPU acceleration
- Support parametric derivatives (vjp, hvp) using automatic differentiation

## Installation

To run the project locally, clone this repository, `cd` into the project folder [`fimjax/`](./) and run

```console
pixi install
```
This installs the environment from the [lock file](./pixi.lock). For further details on how to install and use pixi, see also the [pixi documentation](https://pixi.sh/latest/).


## Usage

An usage example can be found in [`snippets/`](./snippets/).
The main class is the `Solver` class, it supports calculating solutions and Vector-Jacobian-Products.

We use our own wrappers for storing Meshes, the class `Mesh` just needs the vertices of a triangulation as a list of points (M, d) and the triangles as a list of indices (N, 3) into the points array.
Secondly we need a source to solve the Eikonal Equation and a metrics tensor.
The source term is again specified in a wrapper `InitialValues` that takes a list of locations (as indices to the points array) and a list of initial arrival times.
The metrics tensor should be an array (N, d, d) that assigns a constant velocity matrix to the corresponding triangle.
Solving the Eikonal equation on a predefined triangulation consisting of `points` and `elements` with velocities `metrics` can be done as follows:

```py
from fimjax.util.datastructures import Mesh, InitialValues
from fimjax.fimjax import Solver

points = np.array([...])
elements = np.array([...])

mesh = Mesh(points=points, elements=elements)
initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))

iterations = 100
solution = solver.solve(mesh, initial_values, metrics, iterations)
```

For more examples take a look into the [example](snippets/minimal_working_example.py).
By default the solvers uses a fixed number of iterations and AD based on that.
Other methods are implemented, further documentation is found in the code.

## License
This library is licensed under the [GNU Affero General Public License](LICENSE).