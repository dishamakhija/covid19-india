from configs.base_config import TrainingModuleConfig, ForecastingModuleConfig, ModelEvaluatorConfig
from entities.forecast_variables import ForecastVariable
from modules.forecasting_module import ForecastingModule
from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule
from utils.config_util import read_config_file

if __name__ == "__main__":
    ForecastingModule.from_config_file("../config/sample_forecasting_config.json")
    # ModelEvaluator.from_config_file("../config/sample_evaluation_config.json")
    # TrainingModule.from_config_file("../config/sample_training_config.json")
