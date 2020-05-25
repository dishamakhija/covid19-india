import enum


@enum.unique
class InterventionVariable(str, enum.Enum):
    testing_rate = "testing_rate"
    containment_fraction = "containment_fraction"
    mask_compliance = "mask_compliance"


@enum.unique
class InputType(str, enum.Enum):
    param_override = "param_override"
    npi_list = "npi_list"