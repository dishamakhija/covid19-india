from src.configs.base_config import TrainingModuleConfig, ForecastingModuleConfig, ModelEvaluatorConfig
from src.entities.forecast_variables import ForecastVariable
from src.modules.forecasting_module import ForecastingModule
from src.modules.model_evaluator import ModelEvaluator
from src.modules.training_module import TrainingModule
from src.utils.config_util import read_config_file

if __name__ == "__main__":
    # ForecastingModule.from_config_file("/Users/anupama.agarwal/work/covid19-library/config/sample_forecasting_config.json")
    # ModelEvaluator.from_config_file("/Users/anupama.agarwal/work/covid19-library/config/sample_evaluation_config.json")
    TrainingModule.from_config_file("/Users/anupama.agarwal/work/covid19-library/config/sample_training_config.json")
    #
    # config = read_config_file("/Users/anupama.agarwal/work/covid19-library/config/sample_forecasting_config.json")
    # forecasting_module_config = ForecastingModuleConfig.parse_obj(config)
    # print(forecasting_module_config)
    # config = read_config_file("/Users/anupama.agarwal/work/covid19-library/config/sample_evaluation_config.json")
    # evaluation_config = ModelEvaluatorConfig.parse_obj(config)
    # print(evaluation_config)
