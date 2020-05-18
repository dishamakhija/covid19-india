from pydantic import BaseModel
from typing import List, Optional

from entities import InterventionVariable
from entities.forecast_variables import ForecastVariable
from entities.model_class import ModelClass
from entities.loss_function import LossFunction


class BaseConfig(BaseModel):
    region_name: str
    region_type: str
    model_class: ModelClass
    model_parameters: dict
    output_filepath: str = None


class TrainingModuleConfig(BaseConfig):
    train_start_date: str
    train_end_date: str
    search_space: dict
    search_parameters: dict
    training_loss_function: LossFunction
    loss_functions: List[LossFunction]


class ModelEvaluatorConfig(BaseConfig):
    run_day: str
    test_start_date: str
    test_end_date: str
    loss_functions: List[LossFunction]


class ForecastingModuleConfig(BaseConfig):
    run_day: str
    forecast_start_date: str
    forecast_end_date: str
    forecast_variables: List[ForecastVariable]


# Seed Data -  numbers in each compartment (optional)
# Data source to use  (optional)
# List of  ForecastScenarios []
# Start date
# Time intervals []
# End date
# InputType ( ParamOverRide or  NPIList )
# ApplicableInterventions
# Intervention_key_value_pairs []
#
class Intervention(BaseModel):
    intervention_variable: InterventionVariable
    value: float


class ForecastTimeInterval(BaseModel):
    end_date: str
    interventions: List[Intervention]


class ForecastScenario(BaseModel):
    start_date: str
    time_intervals: List[ForecastTimeInterval]


class ScenarioForecastingModuleConfig(BaseConfig):
    run_day: str  # TODO: need to check if it is required
    scenarios: List[ForecastScenario]  ##TODO: does this need to be a list


