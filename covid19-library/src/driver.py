from src.configs.base_config import TrainingModuleConfig, ForecastingModuleConfig, ModelEvaluatorConfig
from src.entities.forecast_variables import ForecastVariable
from src.modules.forecasting_module import ForecastingModule
from src.modules.model_evaluator import ModelEvaluator
from src.modules.training_module import TrainingModule
from src.utils.config_util import read_config_file

if __name__ == "__main__":
    # ForecastingModule.from_config_file("../config/sample_forecasting_config.json")
    # ModelEvaluator.from_config_file("../config/sample_evaluation_config.json")
    TrainingModule.from_config_file("../config/sample_training_config.json")
