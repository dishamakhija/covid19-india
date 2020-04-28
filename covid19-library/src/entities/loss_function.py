from typing import List

from pydantic import BaseModel

from src.entities.forecast_variables import ForecastVariable
from src.entities.metric_name import MetricName


class VariableWeight(BaseModel):
    variable: ForecastVariable
    weight: float


class LossFunction(BaseModel):
    metric_name: MetricName
    variable_weights: List[VariableWeight]
    value: float = 0.0
