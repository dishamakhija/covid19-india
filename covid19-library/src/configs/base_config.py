from pydantic import BaseModel
from typing import List, Optional

from entities.intervention_variable import InputType
from entities.forecast_variables import ForecastVariable
from entities.model_class import ModelClass
from entities.loss_function import LossFunction
from entities.data_source import DataSource


class BaseConfig(BaseModel):
    data_source: str = DataSource.tracker_district_daily
    region_name: List[str]
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


class Intervention(BaseModel):
    intervention_variable: str
    value: float


class ForecastTimeInterval(BaseModel):
    end_date: str
    interventions: List[Intervention]

    def get_interventions_map(self):
        return dict([tuple(intervention.dict().values()) for intervention in self.interventions])


class ScenarioForecastingModuleConfig(BaseConfig):
    run_day: str
    start_date: str
    time_intervals: List[ForecastTimeInterval]
    input_type: InputType
