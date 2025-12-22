import pytest

from pyRadPlan.optimization.solvers import get_available_solvers, get_solver, SolverBase


def test_get_available_solvers():
    solvers = get_available_solvers()
    assert isinstance(solvers, dict)
    assert len(solvers) > 0
    assert all([isinstance(k, str) for k in solvers.keys()])
    assert all([issubclass(v, SolverBase) for v in solvers.values()])


def test_get_solver():
    solvers = get_available_solvers()
    for solver_name, solver_class in solvers.items():
        solver = get_solver(solver_name)
        assert isinstance(solver, solver_class)
        assert isinstance(solver, SolverBase)
        assert solver.short_name == solver_name
        assert solver.name == solver_class.name
        assert solver.short_name == solver_class.short_name


def test_get_solver_invalid():
    with pytest.raises(KeyError):
        get_solver("invalid_solver")
    with pytest.raises(ValueError):
        get_solver(123)
    with pytest.raises(NotImplementedError):
        get_solver({"solver": "scipy"})  # valid solver, but dict not implemented yet
