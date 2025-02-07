"""Integration tests."""

import pytest
import warnings
from fimjax.util.datastructures import Mesh, InitialValues
import numpy as np


def compare_fim_python_vs_fimjax(
    mesh: Mesh, metrics: np.ndarray, initial_values: InitialValues, iterations=500
):
    import numpy as np
    from fimpy.solver import create_fim_solver
    from fimjax.main import Solver
    import logging
    from time import perf_counter

    logger = logging.getLogger(__name__)

    solver = Solver()
    start = perf_counter()
    solution_fimjax = solver.solve(mesh, initial_values, metrics, iterations)
    solution_fimjax.solution.block_until_ready()
    end = perf_counter()
    logger.info(f"fimjax took {end - start}s")

    fim = create_fim_solver(
        mesh.points, mesh.elements, metrics, device="cpu", use_active_list=False
    )
    start = perf_counter()
    solution_fim_python = fim.comp_fim(initial_values.locations, initial_values.values)
    end = perf_counter()
    logger.info(f"fim-python took {end - start}s")

    assert solution_fimjax.has_converged, "Solution did not converge using fimjax"
    assert np.allclose(solution_fim_python, solution_fimjax.solution), (
        "Solution did not match fim-python"
    )


def test_against_fim_python():
    """Integration test against fim-python using Jacobi update."""
    import numpy as np
    from scipy.spatial import Delaunay
    from fimjax.util.datastructures import Mesh, InitialValues
    import jax.numpy as jnp

    # Create triangulated points in 2D
    x = np.linspace(-1, 1, num=50)
    X, Y = np.meshgrid(x, x)
    points = np.stack([X, Y], axis=-1).reshape([-1, 2]).astype(np.float32)
    elems = Delaunay(points).simplices
    elem_centers = np.mean(points[elems], axis=1)
    num_elems = elems.shape[0]
    mesh = Mesh(points=jnp.array(points), elements=jnp.array(elems))
    initial_values = InitialValues(locations=np.array([0]), values=np.array([0]))

    # The domain will have a small spot where movement will be slow
    def velocity_f(x):
        return 1 / (
            1 + np.exp(3.5 - 25 * np.linalg.norm(x - np.array([[0.33, 0.33]]), axis=-1) ** 2)
        )

    velocity_e = velocity_f(elem_centers)  # For computing
    metrics = (
        np.eye(2, dtype=np.float32)[np.newaxis] * velocity_e[..., np.newaxis, np.newaxis]
    )  # Isotropic propagation

    metrics = np.eye(2) * 5
    metrics = np.repeat(metrics[np.newaxis, :, :], num_elems, axis=0)

    compare_fim_python_vs_fimjax(mesh, metrics, initial_values, iterations=300)


def test_heart_against_fim_python():
    """Integration test against fim-python on a 2D-Manifold (real heart)."""
    import numpy as np
    from scipy.spatial import Delaunay
    from fimpy.solver import FIMPY, create_fim_solver
    from fimjax.util.datastructures import Mesh, InitialValues
    import jax.numpy as jnp
    from fimjax.main import Solver
    import logging
    from time import perf_counter

    logger = logging.getLogger(__name__)
    try:
        heart_data = np.load("data/heart.npz")
    except FileNotFoundError:
        warnings.warn("Could not read heart data file.")
        pytest.skip("Could not read heart data file, skipping this test")
        return

    points = heart_data["points"]
    elements = heart_data["elements"]
    x0 = np.array([0])
    x0_vals = np.array([0])
    dim = 3  # dimensionality of points

    mesh = Mesh(points=np.array(points), elements=np.array(elements))
    metrics = np.repeat(np.identity(dim)[np.newaxis, :, :], mesh.elements.shape[0], axis=0)
    initial_values = InitialValues(locations=x0, values=x0_vals)
    compare_fim_python_vs_fimjax(mesh, metrics, initial_values, iterations=300)

    # solver = Solver()
    # start = perf_counter()
    # solution_fimjax = solver.solve(mesh, initial_values, metrics, 300)
    # solution_fimjax.solution.block_until_ready()
    # end = perf_counter()
    # logger.info(f"fimjax took {end-start}s")

    # fim = create_fim_solver(
    #     points, elements, metrics, device="cpu", use_active_list=False
    # )
    # start = perf_counter()
    # solution_fim_python = fim.comp_fim(x0, x0_vals)
    # end = perf_counter()
    # logger.info(f"fim-python took {end-start}s")

    # assert solution_fimjax.has_converged, "Solution did not converge using fimjax"
    # assert np.allclose(
    #     solution_fim_python, solution_fimjax.solution
    # ), "Solution did not match fim-python"


def test_2d_maze():
    """Integration test against fim-python on a 2d maze"""
    import numpy as np
    from fimjax.util.datastructures import InitialValues
    from fimjax.util.mesh_generation import generate_identity_2d_mesh

    mesh, metrics = generate_identity_2d_mesh(50)
    initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))

    # creation of "impermeable" strips in the domain
    indices = []
    disc = np.unique(mesh.points[:, 0])
    yborders = [disc[x] for x in range(5, len(disc), 5)]
    borderxleft = disc[1]
    borderxright = disc[-2]
    tol = 0.001
    for i, e in enumerate(mesh.elements):
        triangle = mesh.points[e]
        for j in range(0, len(yborders), 2):
            if (
                np.any(np.abs(triangle[:, 1] - yborders[j]) < tol)
                and np.all(triangle[:, 1] <= yborders[j])
                and not np.any(np.abs(triangle[:, 0] - borderxright) < tol)
            ):
                indices.append(i)
            if j + 1 >= len(yborders):
                continue
            if (
                np.any(np.abs(triangle[:, 1] - yborders[j + 1]) < tol)
                and np.all(triangle[:, 1] <= yborders[j + 1])
                and not np.any(np.abs(triangle[:, 0] - borderxleft) < tol)
            ):
                indices.append(i)
    metrics = np.array(metrics)
    metrics[np.array(indices)] *= 1e-3  # very slow but still passing fim-pythons assertions

    compare_fim_python_vs_fimjax(mesh, metrics, initial_values, iterations=500)


def test_sphere():
    # golden spiral, see: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    from scipy.spatial import ConvexHull

    n_points = 10000
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=-1)

    hull = ConvexHull(points)
    elements = hull.simplices

    mesh = Mesh(points=points, elements=elements)
    h = 1 * np.identity(3)
    metrics = np.repeat(h[np.newaxis, :, :], mesh.elements.shape[0], axis=0)
    initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))

    compare_fim_python_vs_fimjax(mesh, metrics, initial_values, iterations=500)


if __name__ == "__main__":
    test_sphere()
