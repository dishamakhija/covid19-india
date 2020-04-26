import enum


@enum.unique
class ForecastVariable(str, enum.Enum):
    ##TODO: add definitions
    active = "active"
    recovered = "recovered"
    deceased = "deceased"
    total = "total"
    active_hospitalized = "active_hospitalized"
    active_icu = "active_icu"
    active_ventilation = "active_ventilation"
    active_isolated = "active_isolated"
    active_unknown = "active_unknown"
