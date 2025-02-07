"""This snippet showcases a very simple case where the parametric derivative is not continous.
"""


# %%
from fimjax.util.strenums import ITERATION_SCHEME
from fimjax.util.mesh_generation import generate_identity_2d_mesh
import numpy as np
import jax
import jax.numpy as jnp
from finite_differences import finite_central_difference, finite_forward_difference
from fimjax.util.datastructures import Mesh, InitialValues
from fimjax.main import Solver
from fimjax.core import _update_all_triangles
import matplotlib.pyplot as plt

def plot_triangulation(
    mesh: Mesh,
    points_ind=np.zeros((0,)),
    points_coord=np.zeros((0, 2)),
    triag_ind=np.zeros((0,)),
    triag_coord=np.zeros((0, 3, 2)),
):
    """helper function to plot triangulation.
    """
    points_coord = np.array(points_coord).reshape((-1, 2))
    triag_coord = np.array(triag_coord).reshape((-1, 3, 2))

    points_coord = (
        np.concatenate((points_coord, mesh.points[points_ind]), axis=0)
        if len(points_ind) > 0
        else points_coord
    )
    triag_coord = (
        np.concatenate((triag_coord, mesh.points_triangle[triag_ind]), axis=0)
        if len(triag_ind) > 0
        else triag_coord
    )
    for triangle in triag_coord:
        plt.fill(triangle[:, 0], triangle[:, 1], c="yellow", alpha=0.5)
    plt.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.elements, c="black")

    for i, point in enumerate(points_coord):
        plt.scatter(point[0], point[1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

# %% This is the simplest case where we can see the discontinuity of the FIM solution
# f' is discontinous at M=I due to us picking the minimal path the directional derivatives differ.
if __name__ == '__main__':
    # %%
    solver = Solver()
    solve = solver.get_solver_function(type=ITERATION_SCHEME.FOR, local_update_function=_update_all_triangles)
    mesh, metrics = generate_identity_2d_mesh(2)
    initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))
    def f(alpha):
        alpha1, alpha2 = alpha
        _metrics = jnp.array(metrics.copy())
        _metrics = _metrics.at[0].set(metrics[0]*alpha1)
        _metrics = _metrics.at[1].set(metrics[1]*alpha2)
        return solve(mesh, initial_values, _metrics, 10).solution
    # f = lambda metrics: solve(mesh, initial_values, _metrics, 10).solution

    # adding noise shifts away from the discontinuity making all 3 derivatives agree
    noise = np.random.normal(size=metrics.shape)
    # metrics += 0.01*noise
    plot_triangulation(mesh, points_ind=[0,1,2,3])
    print('Derivative of the solution at [1,1]')
    print('central differences')
    print(finite_central_difference(f, np.array([1,1]), 1e-4)[3])
    print(80*'-')
    print('forward differences')
    print(finite_forward_difference(f, np.array([1,1]), 1e-4)[3])
    print(80*'-')
    print('JAX')
    print(jax.jacfwd(f)(np.array([1,1], dtype=float))[3])

    # %%
    solve(mesh, initial_values, metrics, 3)

# %%
