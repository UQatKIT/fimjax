# Parametric Derivatives

This example describes how to calculate the paramatric derivative $\frac{du(M)}{dM}$ with Fimjax based on algorithmic differentiation implemented by JAX.
In most scenarios we are not interested in the actual Jacobian of $u$, but instead we are interested in minimizing some loss $\ell(u): \mathbb R^N_v \to \mathbb R$, so we want to calculate:

$$
    \frac{d\ell(u(M))}{dM} = \frac{d\ell(u)}{du}\frac{du(M)}{dM}
$$

## Mesh Setup and Solver

The Mesh can be set up similarly as in the example for the [Forward Solver](./solve.md). To make it a bit shorter we use the [generate_identity_2d_mesh][fimjax.util.mesh_generation.generate_identity_2d_mesh] function to shorten the code, which creates a 30x30 mesh together with an isotropic identity tensor.
We then solve this problem as explained in [Forward Solver](./solve.md).

```py
import numpy as np
from fimjax.util.mesh_generation import generate_identity_2d_mesh
from fimjax.util.datastructures import InitialValues
from fimjax.main import Solver

mesh, metrics = generate_identity_2d_mesh(30)
initial_values = InitialValues(locations=np.array([0]), values=np.array([0.]))
solver = Solver()
result = solver.solve(
    mesh = mesh,
    initial_values = initial_values,
    metrics = metrics,
    iter=200
)
```

## Adjoint vector
The first step in calculating $\frac{d\ell(u(M))}{dM}$ is calculating $\frac{d\ell(u)}{du}$.
This can be done analytically for simple loss functions, but we will rely on JAX algorithmic differentiation capabilities instead.

```py
import jax.numpy as jnp
import jax


ground_truth = result.solution
def loss(x):
    return jnp.sum((x-ground_truth)**2)

loss_gradient = jax.grad(loss)
```

We now need a point for which we want to calculate the gradient, we simply choose a different constant speed on the whole domain.

```py
metrics_prime = 2*metrics
solution_prime = solver.solve(
    mesh = mesh,
    initial_values = initial_values,
    metrics = metrics_prime,
    iter=200
).solution

adjoint = loss_gradient(solution_prime)
```

This result gives us the adjoint vector $\frac{d\ell(u)}{du}$.
To get the whole derivative we can now simply call the value_and_vjp (vector-jacobian-product) method of the solver.

```py
val, vjp = solver.value_and_vjp(
    mesh = mesh,
    initial_values = initial_values,
    metrics = metrics_prime,
    iter=200,
    adjoint_vector=adjoint
)
```

This function returns the solution to the eikonal equation as well as  $\frac{d\ell(u(M))}{dM}$,
since the solution needs to be calculated anyways for the derivative.

## Parametrization of the metrics tensor
In the above example the derivative is an array of shape $(1682,2,2)$ corresponding to each entry
in all the different metrics tensors for each triangle.
This is often not the result we are intersted in, instead we often parametrize our metrics tensor with a much smaller number of parameters.
Think of a mapping $M: m \to M$ that maps a number of small parameters to a whole tensor field.
If we are now intersted in the derivative of $l(u)$ w.r.t. $m$ we can simply use the chain rule again to see that:

$$
    \frac{d\ell(u(g(m)))}{dm} = \frac{d\ell(u)}{du}\frac{du(M)}{dM}\frac{dM(m)}{dm}
$$

Note that we have already calculated $\frac{d\ell(u)}{du}\frac{du(M)}{dM}$, which becomes the adjoint vector for $\frac{dM(m)}{dm}$.
We will look at the simple mapping $M: \mathbb R \to \mathbb R^{N_T\times 2\times 2}: M = m \cdot I_2$, where $I_2$ is the identity matrix in 2 dimensions and $M$ is constant on all triangles.
This again corresponds to having a constant isotropic velocity, but we now have this velocity as our one parameter.
We will again use JAX to calculate the VJP w.r.t. to the mapping

```py
m = float(2)
def metric_tensor(m: float):
    return m*metrics

_, vjp_fun = jax.vjp(metric_tensor, m)
dl_dm = vjp_fun(vjp) # 265.15
```
