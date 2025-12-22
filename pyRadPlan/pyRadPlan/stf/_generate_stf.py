from typing import Union, cast
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import (
    SteeringInformation,
    validate_stf,
)
from pyRadPlan.stf.generators import StfGeneratorBase
from pyRadPlan.stf import get_generator


def generate_stf(
    ct: Union[CT, dict], cst: Union[StructureSet, dict], pln: Union[StructureSet, dict]
) -> SteeringInformation:
    ct = validate_ct(ct)
    cst = validate_cst(cst, ct=ct)
    pln = validate_pln(pln)

    stfgen = get_generator(pln)

    stfgen = cast(StfGeneratorBase, stfgen)

    return validate_stf(stfgen.generate(ct, cst))
