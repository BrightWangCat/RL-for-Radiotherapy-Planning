"""Contains all custom exceptions for the stf module."""

from pyRadPlan.core import PyRadPlanError


class GeometryError(PyRadPlanError):
    """Defines an error in the geometry during generation of the stf."""

    def __init__(self, message: str):
        super().__init__(message)
