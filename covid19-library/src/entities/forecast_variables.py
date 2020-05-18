import enum


@enum.unique
class ForecastVariable(str, enum.Enum):
    ##TODO: add definitions
    ## confirmed = active + recovered + deceased
    ## active = hospitalized + icu(no ventilator) + ventilation + isolated + unknown
    active = "active"
    recovered = "recovered"
    deceased = "deceased"
    confirmed = "confirmed"
    hospitalized = "hospitalized"
    icu = "icu"
    ventilation = "ventilation"
    isolated = "isolated"
    unknown = "unknown"
    exposed = "exposed"
    infected = "infected"
    final = "final"
