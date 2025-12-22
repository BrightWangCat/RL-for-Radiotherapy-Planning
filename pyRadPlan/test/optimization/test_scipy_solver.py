import numpy as np

from pyRadPlan.optimization.solvers import get_solver, OptimizerSciPy, SolverBase


def test_get_solver_scipy():
    solver = get_solver("scipy")
    assert isinstance(solver, OptimizerSciPy)
    assert isinstance(solver, SolverBase)
    assert solver.short_name == "scipy"
    assert solver.method == "L-BFGS-B"


def test_simple_problem_scipy():
    solver = get_solver("scipy")

    # Define the problem
    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return [2 * x[0], 2 * x[1]]

    solver.objective = objective
    solver.gradient = gradient

    # Initial guess
    x0 = [1.0, 1.0]

    # Solve
    result = solver.solve(x0)

    assert np.all(np.isclose(result[0], 0.0))
