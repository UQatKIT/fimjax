"""Implementation of several different iteration techniques.

We provide different iteration techniques, because they are handled differently when it comes to
algorithmic differentiation or JAX Just-In-Time compilation.
"""

import jax
from jax import custom_vjp
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit, lax, vjp
from fimjax.util.datastructures import Mesh, InitialValues, FIMSolution
from equinox.internal._loop.checkpointed import checkpointed_while_loop


# Initial solution of every point on the mesh
UNDEFINED_VALUE = 1e10


@partial(jit, static_argnames=["iters", "local_update_function"])
def _compute_fim_for(
    mesh: Mesh,
    initial_values: InitialValues,
    metrics: np.ndarray,
    iters: int,
    local_update_function: callable,
) -> FIMSolution:
    """Uses the Jacobi update with a fixed number of iterations to compute the FIM.

    Uses Checkpointing at every iterations to provide a more memory efficient AD.

    Args:
        mesh: Mesh
        initial_value_locations: position of the initial values in the mesh
        initial_values: initial values
        metrics: [N, d, d] array corresponding to the metric tensor field
        iters: number of iterations
        local_update_function: function that calculates the local updates,
            see _update_all_triangles for more information


    Returns:
        solution along with a flag whether FIM has converged.
    """
    local_update_function = jax.checkpoint(
        local_update_function,
    )
    num_points = mesh.points.shape[0]
    assert mesh.elements.shape[-1] == 3  # currently only triangles are supported
    metrics = jnp.linalg.inv(metrics)
    phi_sol = jnp.ones(shape=(num_points)) * UNDEFINED_VALUE
    phi_sol = phi_sol.at[initial_values.locations].set(initial_values.values)

    u_new = local_update_function(mesh, metrics, phi_sol)

    def loop_body_for(i, state):
        phi_sol, u_new, mesh, metrics, has_converged, has_converged_after = state
        phi_sol = u_new
        u_new = local_update_function(mesh, metrics, phi_sol)
        converged = jnp.allclose(u_new, phi_sol)
        # branchless fuckery for jax
        has_converged_after += (i + 1) * (converged != has_converged)
        has_converged = converged
        return phi_sol, u_new, mesh, metrics, has_converged, has_converged_after

    start = 0
    stop = iters
    has_converged = False
    has_converged_after = -1
    phi_sol, u_new, mesh, metrics, has_converged, has_converged_after = lax.fori_loop(
        start,
        stop,
        loop_body_for,
        (phi_sol, u_new, mesh, metrics, has_converged, has_converged_after),
    )
    solution = FIMSolution(
        solution=phi_sol,
        iterations=iters,
        has_converged=has_converged,
        has_converged_after=has_converged_after,
    )
    return solution


def _compute_fim_while(
    mesh: Mesh,
    initial_values: InitialValues,
    metrics: np.ndarray,
    local_update_function: callable,
) -> FIMSolution:
    """Uses the Jacobi update with a fixed number of iterations to compute the FIM.

    Due to XLA needing static memory bounds this functions is not jittable.

    Args:
        mesh: Mesh
        initial_value_locations: position of the initial values in the mesh
        initial_values: initial values
        metrics: [N, d, d] array corresponding to the metric tensor field
        local_update_function: function that calculates the local updates,
            see _update_all_triangles for more information


    Returns:
        solution object
    """
    num_points = mesh.points.shape[0]
    assert mesh.elements.shape[-1] == 3  # currently only triangles are supported
    metrics = jnp.linalg.inv(metrics)
    phi_sol = jnp.ones(shape=(num_points)) * UNDEFINED_VALUE
    phi_sol = phi_sol.at[initial_values.locations].set(initial_values.values)
    u_new = local_update_function(mesh, metrics, phi_sol)

    def cond_fun(state):
        phi_sol, u_new, mesh, metrics, has_converged_after = state
        return ~jnp.allclose(u_new, phi_sol)

    def body_fun(state):
        phi_sol, u_new, mesh, metrics, has_converged_after = state
        phi_sol = u_new
        has_converged_after += 1
        u_new = local_update_function(mesh, metrics, phi_sol)
        return phi_sol, u_new, mesh, metrics, has_converged_after

    has_converged_after = 0
    phi_sol, u_new, mesh, metrics, has_converged_after = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (phi_sol, u_new, mesh, metrics, has_converged_after),
    )
    solution = FIMSolution(
        solution=phi_sol,
        iterations=has_converged_after,
        has_converged=True,
        has_converged_after=has_converged_after,
    )
    return solution


@partial(jit, static_argnames=["local_update_function", "checkpoints"])
def _compute_fim_checkpointed_while(
    mesh: Mesh,
    initial_values: InitialValues,
    metrics: np.ndarray,
    checkpoints: int,
    local_update_function: callable,
) -> FIMSolution:
    """Uses the Jacobi update until convergence is obtained.

    Uses a checkpointed while loop from equinox to make jitting and AD possible,
    because XLA needs static bounds on memory, that are normally not possible with an unknown number
    of iterations.
    For more information on how this works, please see the documentation of Equinox' checkpointed_while_loop.

    Args:
        mesh: Mesh
        initial_value_locations: position of the initial values in the mesh
        initial_values: initial values
        metrics: [N, d, d] array corresponding to the metric tensor field
        local_update_function: function that calculates the local updates,
            see _update_all_triangles for more information
        checkpoints: number of checkpoints used in forward pass for AD


    Returns:
        solution object
    """
    num_points = mesh.points.shape[0]
    assert mesh.elements.shape[-1] == 3  # currently only triangles are supported
    metrics = jnp.linalg.inv(metrics)
    phi_sol = jnp.ones(shape=(num_points)) * UNDEFINED_VALUE
    phi_sol = phi_sol.at[initial_values.locations].set(initial_values.values)
    u_new = local_update_function(mesh, metrics, phi_sol)

    def cond_fun(state):
        phi_sol, u_new, mesh, metrics, has_converged_after = state
        return ~jnp.allclose(u_new, phi_sol)

    def body_fun(state):
        phi_sol, u_new, mesh, metrics, has_converged_after = state
        phi_sol = u_new
        has_converged_after += 1
        u_new = local_update_function(mesh, metrics, phi_sol)
        return phi_sol, u_new, mesh, metrics, has_converged_after

    has_converged_after = 0
    phi_sol, u_new, mesh, metrics, has_converged_after = checkpointed_while_loop(
        cond_fun,
        body_fun,
        (phi_sol, u_new, mesh, metrics, has_converged_after),
        checkpoints=checkpoints,
    )
    solution = FIMSolution(
        solution=phi_sol,
        iterations=has_converged_after,
        has_converged=True,
        has_converged_after=has_converged_after,
    )
    return solution


# variables
# phi_star is the fixed point solution dependent on the metrics D. So the solution of phi = local_update(phi, D)
# iter and phi_iter are just to extract the iterates from the iteration for plotting
# phi_star_bar is the adjoint vector we want to calculate the vector-jacobian product for
PHI_ITER_COUNT = 1  # static number of iterations to keep for plotting


@jax.tree_util.Partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point(f, D, phi_guess):
    def cond_fun(carry):
        phi_prev, phi, iter, phi_iter = carry
        return ~jnp.allclose(phi_prev, phi, rtol=1e-5, atol=1e-8)  # decide on norm and precision

    def body_fun(carry):
        _, phi, iter, phi_iter = carry
        phi_iter = phi_iter.at[iter].set(phi)
        phi_next = f(D, phi)
        return phi, phi_next, iter + 1, phi_iter

    phi_iter = jnp.zeros((PHI_ITER_COUNT, *phi_guess.shape))
    phi_iter = phi_iter.at[0].set(phi_guess)
    _, phi_star, iter, phi_iter = jax.lax.while_loop(
        cond_fun, body_fun, (phi_guess, f(D, phi_guess), 1, phi_iter)
    )
    return phi_star


def fixed_point_fwd(f, D, phi_init):
    phi_star = fixed_point(f, D, phi_init)
    return phi_star, (D, phi_star)


def fixed_point_rev(f, res, phi_star_bar):
    # see http://implicit-layers-tutorial.org/implicit_functions/
    D, phi_star = res
    _, vjp_D = vjp(lambda a: f(a, phi_star), D)
    _, vjp_phi = vjp(lambda x: f(D, x), phi_star)
    u = fixed_point(lambda _, u: vjp_phi(u)[0] + phi_star_bar, None, jnp.zeros_like(phi_star))
    (D_bar,) = vjp_D(u)
    return D_bar, jnp.zeros_like(phi_star)


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)


# @partial(jit, static_argnames=['local_update_function'])
def _compute_fim_fixed_point(
    mesh: Mesh,
    initial_values: InitialValues,
    metrics: np.ndarray,
    local_update_function: callable,
) -> np.ndarray:
    """Uses the Jacobi update with a fixed number of iterations to compute the FIM.

    Due to XLA needing static memory bounds this functions is not jittable.

    Args:
        mesh: Mesh
        initial_value_locations: position of the initial values in the mesh
        initial_values: initial values
        metrics: [N, d, d] array corresponding to the metric tensor field
        local_update_function: function that calculates the local updates,
            see _update_all_triangles for more information


    Returns:
        solution object
    """
    num_points = mesh.points.shape[0]
    assert mesh.elements.shape[-1] == 3  # currently only triangles are supported
    metrics = jnp.linalg.inv(metrics)
    phi_sol = jnp.ones(shape=(num_points)) * UNDEFINED_VALUE
    phi_sol = phi_sol.at[initial_values.locations].set(initial_values.values)

    return fixed_point(lambda D, phi: local_update_function(mesh, D, phi), metrics, phi_sol)

