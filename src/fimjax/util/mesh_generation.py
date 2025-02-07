"""Helper functions that produce simple 2D rectangular meshes that can be used for testing.
"""

import jax.numpy as jnp
from fimjax.util.datastructures import Mesh
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import ortho_group
import logging


def tensor_field(alpha: float, num_elems: int, dimension: int, dtype=jnp.double) -> np.ndarray:
    """generates identity 

    Args:
        alpha (float): _description_
        num_elems (int): _description_
        dimension (int): _description_
        dtype (_type_, optional): _description_. Defaults to jnp.double.

    Returns:
        np.ndarray: _description_
    """
    D = alpha * jnp.identity(dimension)
    return jnp.repeat(D[jnp.newaxis, :, :], num_elems, axis=0)


def generate_random_2d_mesh(discretization: int) -> tuple[Mesh, np.ndarray]:
    """Generates a random 2D mesh with random metrics (positive definite with eigenvalues between 1 and 5)
    
    Args:
        discretization: Number of points in each dimension
    """
    disc = np.linspace(-1, 1, discretization, dtype=jnp.float64)
    X, Y = np.meshgrid(disc, disc)
    points = np.stack([X, Y], axis=-1).reshape(-1, 2)
    elements = Delaunay(points).simplices
    mesh = Mesh(points=points, elements=elements)
    eigv = ortho_group.rvs(2, size=mesh.elements.shape[0])
    eigh = np.random.uniform(1, 5, size=(mesh.elements.shape[0], 2))
    eigh_matrix = eigh[:, :, np.newaxis] * np.repeat(
        np.identity(2)[np.newaxis, :, :], mesh.elements.shape[0], axis=0
    )  # diagonal tensor
    metrics = np.matmul(eigv, eigh_matrix)
    metrics = np.matmul(metrics, eigv.transpose(0, 2, 1))
    return mesh, metrics


def generate_identity_2d_mesh(discretization: int) -> tuple[Mesh, np.ndarray]:
    """Generates a 2D mesh with isotropic metrics with velocity 1 (identity tensor)

    Args:
        discretization: Number of points in each dimension
    """
    disc = np.linspace(-1, 1, discretization, dtype=jnp.float64)
    X, Y = np.meshgrid(disc, disc)
    points = np.stack([X, Y], axis=-1).reshape(-1, 2)
    elements = Delaunay(points).simplices
    mesh = Mesh(points=points, elements=elements)
    h = 1 * jnp.identity(2)
    metrics = jnp.repeat(h[jnp.newaxis, :, :], mesh.elements.shape[0], axis=0)
    return mesh, metrics


def generate_benchmark_data(dir: str = "data"):
    """Generates benchmark data for benchmarking and also for finite difference validation

    Args:
        dir (optional): directory to store meshes and tensor. Defaults to "data".
    """
    import os
    logger = logging.getLogger(__name__)

    logger.info(f"Generating benchmark data in {dir}")
    os.makedirs(dir, exist_ok=True)
    # larger discretizations
    for discretization in np.arange(10, 801, 10):
        mesh, metrics = generate_random_2d_mesh(discretization)
        np.savez(f"data/random_{discretization}.npz", points=mesh.points, elements=mesh.elements, metrics=metrics, discretization=discretization)
        mesh, metrics = generate_identity_2d_mesh(discretization)
        np.savez(f"data/identity_{discretization}.npz", points=mesh.points, elements=mesh.elements, metrics=metrics, discretization=discretization)

    # small discretizations for finite difference validation
    for discretization in np.arange(3, 16, 1):
        mesh, metrics = generate_random_2d_mesh(discretization)
        np.savez(f"data/random_{discretization}.npz", points=mesh.points, elements=mesh.elements, metrics=metrics, discretization=discretization)
        mesh, metrics = generate_identity_2d_mesh(discretization)
        np.savez(f"data/identity_{discretization}.npz", points=mesh.points, elements=mesh.elements, metrics=metrics, discretization=discretization)


def read_benchmark_data(dir: str = "data", maximal_discretization: int = np.inf) -> tuple[dict[int, tuple[Mesh, np.ndarray]], dict[int, tuple[Mesh, np.ndarray]]]:
    """Reads benchmark data from directory

    Args:
        dir (optional): directory where benchmark data is stored. Defaults to "data".

    Returns:
        dict[int, tuple[Mesh, np.ndarray]]: Dictionary with discretization as key and tuple of mesh and tensor as value
    """
    import os
    logger = logging.getLogger(__name__)
    data_random = dict()
    data_identity = dict()

    logger.info(f"Reading benchmark data from {dir}")
    for filename in os.listdir(dir):
        if filename.endswith(".npz"):
            file = np.load(os.path.join(dir, filename))
            if not 'discretization' in file or not 'points' in file or not 'elements' in file or not 'metrics' in file:
                logger.warning(f"File {filename} does not contain all necessary fields, skipping.")
                continue
            if file['discretization'] <= maximal_discretization:
                if "random" in filename:
                    data_random[int(file['discretization'])] = (Mesh(points=file['points'], elements=file['elements']), file['metrics'])
                elif "identity" in filename:
                    data_identity[int(file['discretization'])] = (Mesh(points=file['points'], elements=file['elements']), file['metrics'])
    return data_random, data_identity


if __name__ == "__main__":
    np.random.seed(0)
    generate_benchmark_data()
