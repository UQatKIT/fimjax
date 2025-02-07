"""A minimal working example showcasing how the Solver class can be used to solve the eikonal equation on different surfaces."""

# %%
from fimjax.util.strenums import ITERATION_SCHEME
from fimjax.util.mesh_generation import generate_identity_2d_mesh
import numpy as np
from fimjax.util.datastructures import Mesh, InitialValues
from fimjax.main import Solver
import matplotlib.pyplot as plt
from time import perf_counter


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


# %% creation of Solver class
solver = Solver()

# %% Generate data
mesh, metrics = generate_identity_2d_mesh(30)
initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))


plot_triangulation(mesh)

start = perf_counter()
sol = solver.solve(mesh, initial_values, metrics, 100)
sol.solution.block_until_ready()
end = perf_counter()
print(
    f"Solution has converged after {sol.has_converged_after} iterations taking {end-start:.2f} seconds"
)

plt.scatter(mesh.points[:, 0], mesh.points[:, 1], c=sol.solution)
plt.colorbar()
plt.title("Pointwise solution")
plt.show()

# %% calculate vjp
adjoint_vec = np.ones(mesh.points.shape[0]) # adjoint vector has the same shape as the solution
start = perf_counter()
primals, vjp = solver.value_and_vjp(mesh, initial_values, metrics, 100, adjoint_vec)
vjp.block_until_ready()
end = perf_counter()
print(
    f"VJP took {end-start:.2f} seconds"
)

# %%
try:
    heart_data = np.load("./data/heart.npz")
except:
    print("Could not read data file")

# %%
points = heart_data["points"]
elements = heart_data["elements"]
dim = 3  # dimensionality of points

mesh = Mesh(points=np.array(points), elements=np.array(elements))
metrics = np.repeat(np.identity(dim)[np.newaxis, :, :], mesh.elements.shape[0], axis=0)
initial_values = InitialValues(locations=np.array([0]), values=np.array([1]))

start = perf_counter()
sol = solver.solve(mesh, initial_values, metrics, 250)
sol.solution.block_until_ready()
end = perf_counter()
print(
    f"Solution has converged after {sol.has_converged_after} iterations taking {end-start:.2f} seconds"
)

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_trisurf(mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2], triangles=mesh.elements)
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
h = ax.scatter(mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2], c=sol.solution)
fig.colorbar(h)
fig.tight_layout()
plt.show()

