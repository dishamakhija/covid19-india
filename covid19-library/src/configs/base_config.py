from pydantic import BaseModel
from typing import List, Optional

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
    add_initial_observation: bool
