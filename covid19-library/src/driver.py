from src.entities.forecast_variables import ForecastVariable
from src.modules.forecasting_module import ForecastingModule
from src.modules.model_evaluator import ModelEvaluator

if __name__ == "__main__":

    # ForecastingModule.from_config("/Users/anupama.agarwal/work/covid-19-library/config/sample_forecasting_config.json")
    ModelEvaluator.from_config("/Users/anupama.agarwal/work/covid-19-library/config/sample_evaluation_config.json")
