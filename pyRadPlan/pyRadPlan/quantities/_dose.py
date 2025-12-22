from numpy.typing import NDArray
import pint

from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class Dose(FluenceDependentQuantity):
    """Dose quantity depending on fluence."""

    unit = ureg.gray
    dim = 1
    identifier = "physical_dose"
    name = "dose"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.identifier not in self._dij.model_fields:
            raise ValueError(f"Quantity {self.identifier} not available in Dij object.")

        # TODO: We could also check if it is an array here

    def _compute_quantity_single_scenario_from_cache(self, scenario_index: int) -> NDArray:
        return self._dij.physical_dose.flat[scenario_index] @ self._w_cache

    def _compute_chained_fluence_gradient_single_scenario_from_cache(
        self, d_quantity, scenario_index: int
    ) -> NDArray:
        return d_quantity @ self._dij.physical_dose.flat[scenario_index]
