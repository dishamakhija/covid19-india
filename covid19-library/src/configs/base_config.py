from pydantic import BaseModel
from typing import List, Optional

from src.entities.forecast_variables import ForecastVariable
from src.entities.model_class import ModelClass
from src.entities.loss_function import LossFunction


class BaseConfig(BaseModel):
    region_name: str
    region_type: str = "district"
    model_class: ModelClass
    model_parameters: dict
    output_filepath: str = None


class TrainingModuleConfig(BaseConfig):
    train_start_date: str
    train_end_date: str
    search_space: dict
    search_parameters: dict
    loss_functions: List[LossFunction]
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None


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
