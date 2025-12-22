"""Dose uniformity."""

from math import sqrt
from numba import njit
from numpy.typing import NDArray

from ._objective import Objective


class DoseUniformity(Objective):
    """Uniformity (minimize standard deviation) objective."""

    name = "Dose Uniformity"

    def compute_objective(self, values):
        return _compute_objective(values)

    def compute_gradient(self, values):
        return _compute_gradient(values)


@njit
def _compute_objective(dose: NDArray):
    return sqrt(len(dose) / (len(dose) - 1)) * dose.std()


@njit
def _compute_gradient(dose: NDArray):
    grad = dose - dose.mean()
    std_val = dose.std()
    if std_val > 0.0:
        grad /= sqrt((len(dose) - 1) * len(dose)) * std_val
    else:
        grad.fill(0.0)

    return grad
