import chex
import numpy as np

@chex.dataclass(init=False)
class Mesh:
    """dataclass for holding everything that belongs to the mesh.
    Note that this only supports keyword arguments due to Chex.
    Use it like:
    `mesh = Mesh(points=points, elements=elements)`

    N: Number of elements in a mesh (i.e. triangles)
    d_e: Number of points in an element
    M: Number of points in a mesh
    d: Dimension of the underlying space

    Attributes:
        points: [M, d] array of points
        elements: [N, d_e] array of indices into points that correspond to elements in a mesh
        points_triangle: [N, d_e, d] like elements, but with the points instead of indices
    """
    points: np.ndarray
    elements: np.ndarray
    points_triangle: np.ndarray

    def __init__(self, points: np.ndarray, elements: np.ndarray):
        """Constructor.

        Args:
            points (np.ndarray): [M, d] array of points
            elements (np.ndarray): [N] array of 
        """
        self.points=points
        self.elements=elements
        self.points_triangle=points[elements]

@chex.dataclass
class InitialValues:
    """Initial values for an Eikonal PDE.

    Attributes:
        locations: [X] array of indices into a mesh that point to the initial value locations
        values: [X] array of values of the initial values
    """
    locations: np.ndarray
    values: np.ndarray


@chex.dataclass
class FIMSolution:
    """Holds the information on a FIMSolution.

    Attributes:
        solution: solution
        iterations: number of iterations
        has_converged: flag if the solution has converged
        has_converged_after: number of iterations needed for convergence
    """
    solution: np.ndarray
    iterations: int
    has_converged: bool
    has_converged_after: int
