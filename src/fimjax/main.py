"""This file contains the main classes that need to be used for solving Eikonal Equations.

The classes from this file should be used as an easy interface to the implemented methods in this
project.
"""

import jax
import numpy as np
from jax import vjp
from fimjax.util.datastructures import Mesh, InitialValues, FIMSolution
from fimjax.util.strenums import ITERATION_SCHEME
from fimjax.core import _update_all_triangles
from fimjax.iterations import (
    _compute_fim_for,
    _compute_fim_checkpointed_while,
    _compute_fim_while,
    _compute_fim_fixed_point,
)


class Solver:
    """Main class to interact with the Eikonal Solver.

    This class should be used as an OOP interface to the JAX code that is used to compute solutions
    and parametric derivatives for the eikonal equation.
    Further information on the different iteration schemes can be found in the corresponding functions.
    """

    def get_solver_function(
        self, type: ITERATION_SCHEME = ITERATION_SCHEME.FOR, local_update_function: callable = None
    ):
        """Provides the solution function as a jax function for composition.

        Returns:
            callable: solution function.
        """
        match type:
            case ITERATION_SCHEME.FOR:
                solver_function = _compute_fim_for
            case ITERATION_SCHEME.WHILE_CHECKPOINTED:
                solver_function = _compute_fim_checkpointed_while
            case ITERATION_SCHEME.WHILE:
                solver_function = _compute_fim_while
            case ITERATION_SCHEME.FOR_FIXED_POINT:
                solver_function = _compute_fim_fixed_point
            case _:
                raise ValueError("Not a valid type")
        if not local_update_function:
            return solver_function
        return jax.tree_util.Partial(solver_function, local_update_function=local_update_function)

    def solve(
        self,
        mesh: Mesh,
        initial_values: InitialValues,
        metrics: np.ndarray,
        iter: int,
        local_update_function: callable = _update_all_triangles,
    ) -> FIMSolution:
        """Uses the FIM algorithm to solve an eikonal equation.

        Args:
            mesh: Mesh
            metrics: metrics tensor
            initial_values: initial values
            iter: number of iterations
            local_update_function (optional): function to use for local updates.
                Defaults to _update_all_triangles.

        Returns:
            FIMSolution: FIMSolution object
        """
        return _compute_fim_for(mesh, initial_values, metrics, iter, local_update_function)

    def value_and_vjp(
        self,
        mesh: Mesh,
        initial_values: InitialValues,
        metrics: np.ndarray,
        iter: int,
        adjoint_vector: np.ndarray,
        local_update_function: callable = _update_all_triangles,
    ) -> np.ndarray:
        """Calculates the value and the vector-jacobian product for the FIM.

        Args:
            mesh: Mesh
            metrics: metrics tensor
            initial_values: initial values
            iter: number of iterations
            adjoint_vector: adjoint vector
            local_update_function (optional): function to use for local updates.
                Defaults to _update_all_triangles.

        Returns:
            np.ndarray: adjoint vector
        """
        fim_fun = lambda metric: _compute_fim_for(
            mesh, initial_values, metric, iter, local_update_function
        ).solution
        primals_out, vjp_fun = vjp(fim_fun, metrics)
        return primals_out, vjp_fun(adjoint_vector)[0]
