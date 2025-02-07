"""Comparison of the parametric derivative from JAX algorithmic differentiation and finite differences.
"""

# %%
import jax
from fimjax.util.datastructures import InitialValues
from fimjax.core import _update_all_triangles
from fimjax.main import Solver
from fimjax.util.strenums import ITERATION_SCHEME
from fimjax.util.mesh_generation import read_benchmark_data
import numpy as np
import matplotlib.pyplot as plt
import logging

# this forces double precision on CPU.
# Running on GPU might lead to different results if single precision is used
jax.config.update("jax_enable_x64", True)


def finite_central_difference(f, x, h):
    x_flat = np.array(x, dtype=float).flatten()
    jacobian = []
    for i in range(x_flat.shape[0]):
        _h = np.zeros(x_flat.shape[0])
        _h[i] = h
        _h = _h.reshape(x.shape)
        jacobian.append((f(x + _h) - f(x - _h)) / (2 * h))
    return np.array(jacobian).T.reshape(-1, *x.shape)


def finite_forward_difference(f, x, h):
    x_flat = np.array(x, dtype=float).flatten()
    jacobian = []
    for i in range(x_flat.shape[0]):
        _h = np.zeros(x_flat.shape[0])
        _h[i] = h
        _h = _h.reshape(x.shape)
        jacobian.append((f(x + _h) - f(x)) / h)
    return np.array(jacobian).T.reshape(-1, *x.shape)


logging.basicConfig(level=logging.INFO)
# %% CALCULATIONS
if __name__ == "__main__":
    # %%
    logger = logging.getLogger(__name__)
    with open("finite_difference_comparison.txt", "w") as file:
        file.write(
            "central FD relative error in Frobenius norm, forward FD relative error in Frobenius norm, degrees of freedom, finite difference step size\n"
        )
    data_random, data_identity = read_benchmark_data(
        "data/", maximal_discretization=15
    )  # larger meshes are too slow for fd
    initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))

    solver = Solver()
    solve = solver.get_solver_function(
        type=ITERATION_SCHEME.FOR, local_update_function=_update_all_triangles
    )

    for discretization, (mesh, metrics) in data_random.items():
        logger.info(f"Calculating errors for mesh with {mesh.elements.shape[0]} degrees of freedom")
        solver = Solver()
        solve = solver.get_solver_function(
            type=ITERATION_SCHEME.FOR, local_update_function=_update_all_triangles
        )

        f = lambda metrics: solve(mesh, initial_values, metrics, 100).solution
        parameter_vector = metrics
        jacobian_jax = jax.jacrev(f)(parameter_vector)
        jacobian_jax_matrix = jacobian_jax.reshape(mesh.points.shape[0], -1)
        hs = np.logspace(-1, -8, 20)
        for h in hs:
            jacobian_central_differences = finite_central_difference(f, parameter_vector, h)
            jacobian_central_differences_matrix = jacobian_central_differences.reshape(
                mesh.points.shape[0], -1
            )
            jacobian_forward_differences = finite_forward_difference(f, parameter_vector, h)
            jacobian_forward_differences_matrix = jacobian_forward_differences.reshape(
                mesh.points.shape[0], -1
            )
            error_central_fd = np.linalg.norm(
                jacobian_jax_matrix - jacobian_central_differences_matrix
            )
            error_forward_fd = np.linalg.norm(
                jacobian_jax_matrix - jacobian_forward_differences_matrix
            )
            with open("finite_difference_comparison.txt", "a") as file:
                file.write(
                    f"{error_central_fd/np.linalg.norm(jacobian_central_differences_matrix)}, {error_forward_fd/np.linalg.norm(jacobian_forward_differences_matrix)}, {mesh.elements.shape[0]}, {h}\n"
                )


# %% PLOTTING
if __name__ == "__main__":
    # %%
    loggger = logging.getLogger(__name__)
    with open("finite_difference_comparison.txt", "r") as file:
        data = file.read()
    data = data.split("\n")[1:-1]
    data_dict = dict()
    for line in data:
        error_central_fd, error_forward_fd, deg_freedom, h = line.split(",")
        error_central_fd = float(error_central_fd)
        error_forward_fd = float(error_forward_fd)
        deg_freedom = int(deg_freedom)
        h = float(h)
        if deg_freedom in data_dict:
            data_dict[deg_freedom] = (
                *data_dict[deg_freedom],
                (h, error_central_fd, error_forward_fd),
            )
        else:
            data_dict[deg_freedom] = ((h, error_central_fd, error_forward_fd),)
    logger.info(f"Found data for the following mesh sizes (number of triangles) {data_dict.keys()}")

    for deg_freedom in (8, 50, 128, 392):
        hs = np.array(data_dict[deg_freedom])[:, 0]
        error_central_fd = np.array(data_dict[deg_freedom])[:, 1]
        plt.loglog(hs, error_central_fd, label=deg_freedom)
    plt.gca().set_prop_cycle(None)
    for deg_freedom in (8, 50, 128, 392):
        hs = np.array(data_dict[deg_freedom])[:, 0]
        error_forward_fd = np.array(data_dict[deg_freedom])[:, 2]
        plt.loglog(hs, error_forward_fd, label=deg_freedom, linestyle="dashed")
    plt.legend()
    plt.xlabel("finite difference step size")
    plt.ylabel("$\|\mathcal{J}_{FD} - \mathcal{J}_{JAX}\|_F / \|\mathcal{J}_{FD}\|_F$")
    plt.gca().invert_xaxis()
    # plt.title('finite difference comparison for different number of degrees\n of freedom for central FD (solid) and forward FD (dashed)')
    plt.tight_layout()
    plt.savefig("finite_difference_comparison.pdf")
# %%
