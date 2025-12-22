from pyRadPlan.optimization.problems import (
    NonLinearFluencePlanningProblem,
    NonLinearPlanningProblem,
    PlanningProblem,
)
from pyRadPlan.plan import create_pln


def test_construct():
    pln = create_pln({"radiation_mode": "protons", "machine": "Generic"})
    prob = NonLinearFluencePlanningProblem(pln)

    assert isinstance(prob, NonLinearFluencePlanningProblem)
    assert isinstance(prob, NonLinearPlanningProblem)
    assert isinstance(prob, PlanningProblem)


def test_construct_noplan():
    prob = NonLinearFluencePlanningProblem()

    assert isinstance(prob, NonLinearFluencePlanningProblem)
    assert isinstance(prob, NonLinearPlanningProblem)
    assert isinstance(prob, PlanningProblem)
