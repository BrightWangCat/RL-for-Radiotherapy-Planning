"""Beamlet datamodels for particle and photon beamlets."""

from typing import Any
from pydantic import Field, field_serializer, SerializerFunctionWrapHandler, FieldSerializationInfo
import numpy as np
from pyRadPlan.stf._rangeshifter import RangeShifter
from pyRadPlan.core import PyRadPlanBaseModel

# TODO: We need to figure out how we can do nested validation in pydantic and then differentiate
# between photon and particle beamlets. For now, we will use the Beamlet class for both.


class Beamlet(PyRadPlanBaseModel):
    """
    A class representing a single beamlet.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the bemamlet information, including properties like
    energy & monitor units.

    Attributes
    ----------
    energy : float
        The energy value for the beamlet
    num_particles_per_mu : float
        The number of particles per monitor unit
    relative_fluence : float
        The fluence of this beamlet relative to the central primary fluence.
        For example, due to a non-uniform primary fluence
    weight : float
        The applied fluence weight of the beamlet
    min_mu : float
        The minimum monitor unit
    max_mu : float
        The maximum monitor unit
    range_shifter : RangeShifter
        The range shifter applied for the beamlet.
    focus_ix : int
        The focus index identifying the focus setting for the beamlet.
    """

    energy: float
    num_particles_per_mu: float = Field(alias="numParticlesPerMU", default=1.0e6)
    min_mu: float = Field(alias="minMU", default=0.0)
    max_mu: float = Field(alias="maxMU", default=float("inf"))
    relative_fluence: float = Field(default=1.0)
    weight: float = Field(default=1.0)
    range_shifter: RangeShifter = Field(default_factory=RangeShifter)
    focus_ix: int = Field(default=0)

    @field_serializer(
        "energy",
        "num_particles_per_mu",
        "min_mu",
        "max_mu",
        "relative_fluence",
        "weight",
        "focus_ix",
        mode="wrap",
    )
    def field_typing(
        self, v: Any, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> Any:
        """Ensure correct serialization in various contexts."""

        context = info.context
        if context and context.get("matRad") == "mat-file":
            if info.field_name == "focus_ix":
                return np.float64(v + 1)  # Increment focus_ix by 1 for MATLAB/matRad
            return np.float64(v)  # Ensure double for MATLAB/matRad

        return handler(v, info)


# Do not use yet


class IonSpot(Beamlet):
    """
    A class representing a single beamlet.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information specific to particles, including properties like
    range shifter and focus index.

    Attributes
    ----------
    range_shifter : RangeShifter
        The range shifter applied for the beamlet.
    focus_ix : int
        The focus index identifying the focus setting for the beamlet.
    """


class PhotonBixel(Beamlet):
    """
    A class representing a single photon beamlet.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information for photons. Mainly the relative fluence of the beamlet /
    bixel due to its lateral position is stored in here.

    Attributes
    ----------
    relative_fluence : float
        The fluence of this beamlet relative to the central primary fluence.
    """


# Example
if __name__ == "__main__":
    testbeamlet = Beamlet(energy=100)
