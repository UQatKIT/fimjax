"""Script to produce the runtime plots seen in the `/plots` folder.

Note that the meshes have to be generated first , so please execute `./snippets/mesh_generation.py`
first to ensure that the meshes are generated.
"""

# %%
import jax
from fimjax.util.datastructures import Mesh, InitialValues
from fimjax.main import Solver
from fimjax.core import _update_all_triangles
from fimjax.util.strenums import ITERATION_SCHEME
from fimjax.util.mesh_generation import read_benchmark_data
import numpy as np
import time
import matplotlib.pyplot as plt
import logging


def runtime_solve(
    mesh: Mesh,
    metrics: np.ndarray,
    initial_values: InitialValues,
    num_trials: int = 10,
    iterations_multiplier: int = 3,
    device: str = "cpu",
):
    """Benchmark runtime of fimjax for a given mesh and metric.

    Args:
        mesh: mesh
        metrics: metrics data
        initial_values: initial values
        num_trials (optional): number of trials to run. Defaults to 10.
        iterations_multiplier (optional): multiplier that is used to set the number of initial iterations. Defaults to 3.
        device (optional): device to run the benchmark on either 'cpu' or 'gpu'. Defaults to 'cpu'.
    """
    logger = logging.getLogger(__name__)
    try:
        gpu_device = jax.devices("gpu")[0]
        cpu_device = jax.devices("cpu")[0]
    except RuntimeError:
        if device == "gpu":
            logger.error("No GPU found, cannot generate benchmark data for GPU.")
            return None, None
        cpu_device = jax.devices("cpu")[0]

    jax_device = cpu_device if device == "cpu" else gpu_device
    with jax.default_device(jax_device):
        solver = Solver()
        solve = solver.get_solver_function(
            type=ITERATION_SCHEME.FOR, local_update_function=_update_all_triangles
        )
        sol = solve(
            mesh,
            initial_values,
            metrics,
            int(np.sqrt(mesh.elements.shape[0]) * iterations_multiplier),
        )
        iterations = int(sol.has_converged_after)
        logger.debug(
            f"Testrun of mesh with {mesh.elements.shape[0]} degrees of freedom on {sol.solution.device} device converged after {iterations} iterations"
        )
        if iterations == -1:
            logger.error(
                "Warning: solution did not converge, skipping this trial. To fix this, try to increase iterations_multiplier"
            )
            return None, None

        sol = solve(
            mesh, initial_values, metrics, iterations
        ).solution.block_until_ready()  # solve once to ensure jit compilation is done
        (
            jax.device_put(0.0, device=jax_device) + 0
        ).block_until_ready()  # make sure device is ready
        runtimes = []
        for _ in range(num_trials):
            start = time.perf_counter()
            sol = solve(
                mesh, initial_values, metrics, iterations
            ).solution.block_until_ready()  # this run should be without jit compilation time
            end = time.perf_counter()
            runtimes.append(end - start)
    return runtimes, mesh.elements.shape[0]


def runtime_derivative(
    mesh: Mesh,
    metrics: np.ndarray,
    initial_values: InitialValues,
    num_trials: int = 10,
    iterations_multiplier: int = 3,
    device: str = "cpu",
):
    """Benchmark runtime of a simple vjp of fimjax for a given mesh and metric.

    Args:
        mesh: mesh
        metrics: metrics data
        initial_values: initial values
        num_trials (optional): number of trials to run. Defaults to 10.
        iterations_multiplier (optional): multiplier that is used to set the number of initial iterations. Defaults to 3.
        device (optional): device to run the benchmark on either 'cpu' or 'gpu'. Defaults to 'cpu'.
    """
    logger = logging.getLogger(__name__)
    try:
        gpu_device = jax.devices("gpu")[0]
        cpu_device = jax.devices("cpu")[0]
    except RuntimeError:
        if device == "gpu":
            logger.error("No GPU found, cannot generate benchmark data for GPU.")
            return None, None
        cpu_device = jax.devices("cpu")[0]

    jax_device = cpu_device if device == "cpu" else gpu_device

    with jax.default_device(jax_device):
        solver = Solver()
        solve = solver.get_solver_function(
            type=ITERATION_SCHEME.FOR, local_update_function=_update_all_triangles
        )
        solve = jax.jit(solve, device=jax_device, static_argnums=(3,))
        sol = solve(
            mesh,
            initial_values,
            metrics,
            int(np.sqrt(mesh.elements.shape[0]) * iterations_multiplier),
        )
        iterations = int(sol.has_converged_after)

        logger.debug(
            f"Testrun of mesh with {mesh.elements.shape[0]} degrees of freedom on {sol.solution.device} device converged after {iterations} iterations"
        )
        if iterations == -1:
            logger.error(
                "Warning: solution did not converge, skipping this trial. To fix this, try to increase iterations_multiplier"
            )
            return None, None

        def simple_loss(metrics, mesh, initial_values):
            sol = solve(mesh, initial_values, metrics, iterations)
            return np.sum(sol.solution)

        jitted_grad = jax.jit(jax.grad(simple_loss))
        gradient = jitted_grad(
            metrics, mesh, initial_values
        ).block_until_ready()  # run once to ensure jit compilation
        (
            jax.device_put(0.0, device=jax_device) + 0
        ).block_until_ready()  # make sure device is ready

        runtimes = []
        for _ in range(num_trials):
            start = time.perf_counter()
            gradient = jitted_grad(
                metrics, mesh, initial_values
            ).block_until_ready()  # this run should be without jit compilation time
            end = time.perf_counter()
            runtimes.append(end - start)
    return runtimes, mesh.elements.shape[0]


# %%
if __name__ == "__main__":
    # %%
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("jax").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    import os

    if "data" not in os.listdir("."):
        logger.error(
            "No data directory found. Please make sure to run this out of the root directory."
        )
        raise FileNotFoundError(
            "No data directory found. Please make sure to run this out of the root directory."
        )
    # %% benchmark eikonal solver
    data_random, data_identity = read_benchmark_data("./data")
    initial_values = InitialValues(locations=np.array([0]), values=np.array([0]))
    runtimes_gpu, degrees_of_freedom_gpu = [], []
    runtimes_cpu, degrees_of_freedom_cpu = [], []
    with open("runtimes_eikonal_solver.txt", "w") as f:
        f.write("degrees_of_freedom_gpu; runtimes_gpu; degrees_of_freedom_cpu; runtimes_cpu\n")

    for disc, (mesh, metrics) in sorted(data_random.items()):
        if (disc >= 400) and (disc % 50 != 0):
            continue
        logger.info(
            f"Runing solve benchmark for mesh with {mesh.elements.shape[0]} degrees of freedom"
        )
        runtime_cpu, degree_of_freedom_cpu = runtime_solve(
            mesh, metrics, initial_values, device="cpu"
        )
        runtime_gpu, degree_of_freedom_gpu = runtime_solve(
            mesh, metrics, initial_values, device="gpu"
        )
        with open("runtimes_eikonal_solver.txt", "a") as f:
            f.write(
                f"{degree_of_freedom_gpu};{runtime_gpu};{degree_of_freedom_cpu};{runtime_cpu}\n"
            )

    logging.info("Finished benchmarking fimjax")
    del data_random
    del data_identity

    # %% benchmark black box automatic differentiation for eikonal solver
    data_random, data_identity = read_benchmark_data("./data")
    initial_values = InitialValues(locations=np.array([0]), values=np.array([0]))
    runtimes_gpu, degrees_of_freedom_gpu = [], []
    runtimes_cpu, degrees_of_freedom_cpu = [], []
    with open("runtimes_derivative_eikonal_solver.txt", "w") as f:
        f.write("degrees_of_freedom_gpu; runtimes_gpu; degrees_of_freedom_cpu; runtimes_cpu\n")

    for disc, (mesh, metrics) in sorted(data_random.items()):
        if (disc >= 400) and (disc % 50 != 0):
            continue
        logger.info(
            f"Runing gradient benchmark for mesh with {mesh.elements.shape[0]} degrees of freedom"
        )
        runtime_cpu, degree_of_freedom_cpu = runtime_derivative(
            mesh, metrics, initial_values, device="cpu"
        )
        runtime_gpu, degree_of_freedom_gpu = runtime_derivative(
            mesh, metrics, initial_values, device="gpu"
        )
        with open("runtimes_derivative_eikonal_solver.txt", "a") as f:
            f.write(
                f"{degree_of_freedom_gpu};{runtime_gpu};{degree_of_freedom_cpu};{runtime_cpu}\n"
            )

    logging.info("Finished benchmarking derivative of fimjax")
    del data_random
    del data_identity

    # %% plot runtimes
    plt.clf()
    logger.info("Plotting runtimes")
    with open("runtimes_eikonal_solver.txt") as f:
        lines = f.readlines()
        degrees_of_freedom_gpu = []
        runtimes_gpu = []
        degrees_of_freedom_cpu = []
        runtimes_cpu = []
        for line in lines[1:]:
            values = [v.strip() for v in line.split(";")]
            degrees_of_freedom_gpu.append(int(values[0]))
            runtime_gpu = [float(x) for x in values[1][1:-1].split(",")]
            runtimes_gpu.append(np.mean(runtime_gpu))
            degrees_of_freedom_cpu.append(int(values[2]))
            runtime_cpu = [float(x) for x in values[3][1:-1].split(",")]
            runtimes_cpu.append(np.mean(runtime_cpu))

    i = np.argsort(degrees_of_freedom_cpu)
    plt.loglog(np.array(degrees_of_freedom_gpu)[i], np.array(runtimes_gpu)[i], label="GPU")
    plt.loglog(np.array(degrees_of_freedom_cpu)[i], np.array(runtimes_cpu)[i], label="CPU")
    plt.xlabel("Degrees of freedom")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.title("Runtime of eikonal solver")
    plt.grid(True, which="both")
    plt.savefig("runtimes_eikonal_solver.pdf")
    plt.show()

    # %% plot derivative runtimes
    plt.clf()
    logger.info("Plotting derivative runtimes")
    with open("runtimes_derivative_eikonal_solver.txt") as f:
        lines = f.readlines()
        degrees_of_freedom_gpu = []
        runtimes_gpu = []
        degrees_of_freedom_cpu = []
        runtimes_cpu = []
        for line in lines[1:]:
            values = [v.strip() for v in line.split(";")]
            degrees_of_freedom_gpu.append(int(values[0]))
            runtime_gpu = [float(x) for x in values[1][1:-1].split(",")]
            runtimes_gpu.append(np.mean(runtime_gpu))
            degrees_of_freedom_cpu.append(int(values[2]))
            runtime_cpu = [float(x) for x in values[3][1:-1].split(",")]
            runtimes_cpu.append(np.mean(runtime_cpu))

    i = np.argsort(degrees_of_freedom_cpu)
    plt.loglog(np.array(degrees_of_freedom_gpu)[i], np.array(runtimes_gpu)[i], label="GPU")
    plt.loglog(np.array(degrees_of_freedom_cpu)[i], np.array(runtimes_cpu)[i], label="CPU")
    plt.xlabel("Degrees of freedom")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.title("Runtime of eikonal solver derivative via loop-unrolling")
    plt.grid(True, which="both")
    plt.savefig("runtimes_derivative_eikonal_solver.pdf")
# %%
