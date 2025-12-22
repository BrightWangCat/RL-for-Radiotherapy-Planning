"""Mean dose objective."""

from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import ones

from ._objective import Objective, ParameterMetadata

# %% Class definition


class MeanDose(Objective):
    """
    Mean Dose objective.

    Attributes
    ----------
    d_ref : float
        referene mean dose to achieve

    Notes
    -----
    While we implement a reference value, we suggest to only use 0 as reference
    """

    name = "Mean Dose"

    d_ref: Annotated[float, Field(default=0.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values):
        return _compute_objective(values, self.d_ref)

    def compute_gradient(self, values):
        return _compute_gradient(
            values,
            self.d_ref,
        )


@njit
def _compute_objective(dose, d_ref):
    return (dose.mean() - d_ref) ** 2


@njit
def _compute_gradient(dose, d_ref):
    grad = 2 * (dose.mean() - d_ref) * ones(dose.shape) / dose.size
    return grad
