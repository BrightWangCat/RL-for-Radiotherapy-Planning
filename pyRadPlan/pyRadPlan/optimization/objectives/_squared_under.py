"""Squared Underdosing Objective."""

from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import clip

from ._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredUnderdosing(Objective):
    """
    Squared Underdosing (piece-wise negative least-squares) objective.

    Attributes
    ----------
    d_min : float
        minimum dose value (below which we penalize)
    """

    name = "Squared Underdosing"

    d_min: Annotated[float, Field(default=60.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values):
        return _compute_objective(values, self.d_min)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.d_min)


@njit
def _compute_objective(dose, d_min):
    underdose = clip(dose - d_min, a_min=None, a_max=0)

    return (underdose @ underdose) / len(dose)


@njit
def _compute_gradient(dose, d_min):
    underdose = clip(dose - d_min, a_min=None, a_max=0)
    return 2.0 * underdose / len(underdose)
