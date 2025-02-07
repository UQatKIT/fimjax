"""This file contains the core functions needed for the local update.

For readability we sometimes include the shape of arrays, with the following meaning:
N: number of elements in mesh (i.e. triangles)
d_e: number of points in one element
M: number of points in mesh
d: dimension of the underlying space
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from fimjax.util.datastructures import Mesh


def _norm_squared(A: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    r"""Custom vectorized implementation of :math:`\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>`.

    Args:
        A: [N, d, d] Array of tensors
        x1: [N, d] Point 1
        x2: [N, d] Point 2

    Returns:
        [N] array of norm values
    """
    vectorized_dot = jax.vmap(lambda A, x1, x2: jnp.dot(A @ x1, x2))
    return vectorized_dot(A, x1, x2)


def _norm(A: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    r"""Custom vectorized implementation of:math:`\\sqrt{\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>}`.

    Args:
        A: [N, d, d] Array of tensors
        x1: [N, d] Point 1
        x2: [N, d] Point 2

    Returns:
        [N] array of norm values
    """
    squared_norm = _norm_squared(A, x1, x2)
    squared_norm = jnp.where(
        squared_norm >= 0, squared_norm, 0
    )  # don't calculate negative square roots, otherwise AD fails
    return jnp.sqrt(squared_norm)


def _update_point(
    x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, D: np.ndarray, u1: np.ndarray, u2: np.ndarray
) -> np.ndarray:
    """Calculates the minimal solution along the border of a triangle.

    Args:
        x1: [N, d] array with stack of the first vertex of the triangles
        x2: [N, d] array with stack of the second vertex of the triangles
        x3: [N, d] array with stack of the third vertex of the triangles
        D: [N, d] array with stack of speed tensors
        u1: [N] array with current solution at the first vertex
        u2: [N] array with current solution at the first vertex

    Returns:
        [N] array minimal solution along the borders of the triangle
    """
    a1 = x3 - x1
    a2 = x3 - x2
    u3_1 = u1 + _norm(D, a1, a1)
    u3_2 = u2 + _norm(D, a2, a2)
    return jnp.minimum(u3_1, u3_2)


def _update_triangle(
    x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, D: np.ndarray, u1: np.ndarray, u2: np.ndarray
) -> np.ndarray:
    """Update triangle(s) locally.

    Calculates all triangle updates in a broadcasted way
    by solving a constrained optimization problem analytically.
    See paper on FIM for triangulated meshes for more infos.

    Args:
        x1: [N, d] array with stack of the first vertex of the triangles
        x2: [N, d] array with stack of the second vertex of the triangles
        x3: [N, d] array with stack of the third vertex of the triangles
        D: [N, d] array with stack of speed tensors
        u1: [N] array with current solution at the first vertex
        u2: [N] array with current solution at the first vertex

    Returns:
        [N] minimal solution along the borders of the triangle
    """
    k = u1 - u2
    z2 = x2 - x3
    z1 = x1 - x2

    p11 = _norm_squared(D, x1=z1, x2=z1)
    p12 = _norm_squared(D, x1=z1, x2=z2)
    p22 = _norm_squared(D, x1=z2, x2=z2)
    denominator = p11 - k**2
    denominator = jnp.where(denominator > 0, denominator, 0.0001)
    sqrt_val = jnp.where(denominator > 0, (p11 * p22 - p12**2) / denominator, jnp.inf)
    sqrt_invalid_mask = sqrt_val <= 0.0
    sqrt_valid_mask = sqrt_val > 0.0
    # sqrt_op = jnp.sqrt(sqrt_val)
    sqrt_val = jnp.where(sqrt_valid_mask, sqrt_val, jnp.inf)
    sqrt_op = jnp.where(sqrt_valid_mask, jnp.sqrt(sqrt_val), jnp.inf)
    # for the double-where trick read: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    rhs = k * sqrt_op
    alpha1 = -(p12 + rhs) / p11
    alpha2 = -(p12 - rhs) / p11
    alpha1 = jnp.minimum(jnp.maximum(alpha1, 0.0), 1.0)
    alpha2 = jnp.minimum(jnp.maximum(alpha2, 0.0), 1.0)

    u3 = []
    for alpha in [alpha1, alpha2]:
        x = x3 - (alpha[..., jnp.newaxis] * x1 + (1 - alpha[..., jnp.newaxis]) * x2)
        u3.append(alpha * u1 + (1 - alpha) * u2 + _norm(D, x, x))

    u3 = jnp.minimum(*u3)
    u3_point = _update_point(x1, x2, x3, D, u1, u2)
    u3_computed = jnp.where(sqrt_invalid_mask, u3_point, u3)
    u3_final = u3_computed

    return u3_final


@jit
def _update_all_triangles(
    mesh: Mesh,
    D: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    """Performs one Jacobi update.

    Calculates the solution to all update direction and picks the smallest one for each point.

    Args:
        mesh: Mesh
        D: [N, d, d] array with metric tensor field
        solution: [M] array solution before iteration

    Returns:
        [M] new solution after one iteration
    """
    solution_new = solution.copy()
    # updating from each "direction"
    for permutation in np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]]):
        triangles = mesh.elements.at[:, permutation].get()
        points = mesh.points_triangle.at[:, permutation].get()
        solution_at_triangles = solution.at[triangles].get()
        solution_updated = _update_triangle(
            x1=points.at[:, 0].get(),
            x2=points.at[:, 1].get(),
            x3=points.at[:, 2].get(),
            D=D,
            u1=solution_at_triangles[:, 0],
            u2=solution_at_triangles[:, 1],
        )

        reshaped_solution_new = solution_new.at[triangles[:, -1]].get()
        mins = jnp.minimum(reshaped_solution_new, solution_updated)
        solution_new = solution_new.at[triangles[:, -1]].min(mins)
    return solution_new


def _update_all_triangles_no_self_update(
    mesh: Mesh,
    D: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    """Performs one Jacobi update.

    Calculates the solution to all update direction and picks the smallest one for each point.
    This does not include the self update, so a node is always updated from the adjacent nodes,
    so using this method the iteration may NOT converge.

    Args:
        mesh: Mesh
        D: [N, d, d] array with metric tensor field
        solution: [M] array solution before iteration

    Returns:
        [M] new solution after one iteration
    """
    solution_new = solution.copy()
    # updating from each "direction"
    for permutation in np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]]):
        triangles = mesh.elements.at[:, permutation].get()
        points = mesh.points_triangle.at[:, permutation].get()
        solution_at_triangles = solution.at[triangles].get()
        solution_updated = _update_triangle(
            x1=points.at[:, 0].get(),
            x2=points.at[:, 1].get(),
            x3=points.at[:, 2].get(),
            D=D,
            u1=solution_at_triangles[:, 0],
            u2=solution_at_triangles[:, 1],
        )

        reshaped_solution_new = solution_new.at[triangles[:, -1]].get()
        mins = jnp.minimum(reshaped_solution_new, solution_updated)
        solution_new = solution_new.at[triangles[:, -1]].set(mins)
    return solution_new


def _update_all_triangles_softmin(
    mesh: Mesh,
    D: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    """Performs one Jacobi update.

    Calculates the solution to all update direction and picks the smallest one for each point.
    Instead of the minimum the softmin is used.

    Args:
        mesh: Mesh
        D: [N, d, d] array with metric tensor field
        solution: [M] array solution before iteration

    Returns:
        [M] new solution after one iteration
    """
    solution_new = solution.copy()
    # updating from each "direction"
    for permutation in np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]]):
        triangles = mesh.elements.at[:, permutation].get()
        points = mesh.points_triangle.at[:, permutation].get()
        solution_at_triangles = solution.at[triangles].get()
        solution_updated = _update_triangle(
            x1=points.at[:, 0].get(),
            x2=points.at[:, 1].get(),
            x3=points.at[:, 2].get(),
            D=D,
            u1=solution_at_triangles[:, 0],
            u2=solution_at_triangles[:, 1],
        )

        reshaped_solution_new = solution_new.at[triangles[:, -1]].get()
        h1 = jnp.exp(-reshaped_solution_new)
        h2 = jnp.exp(-solution_updated)
        s = h1 + h2
        mins = reshaped_solution_new * (h1 / s) + solution_updated * (h2 / s)
        solution_new = solution_new.at[triangles[:, -1]].min(mins)
    return solution_new
