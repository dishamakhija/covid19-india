from src.model_wrappers.model_factory import ModelFactory
from src.modules.forecasting_module import ForecastingModule
from src.utils import metrics_util
import json

from src.utils.config_util import read_config_file


class ModelEvaluator(object):

    def __init__(self, model_class, model_parameters):
        self._model = ModelFactory.get_model(model_class, model_parameters)
        self._forecasting_module = ForecastingModule(model_class, model_parameters)

    def evaluate(self, region_name, run_day, test_start_date, test_end_date, loss_functions, output_filepath=None):
        predictions = self._forecasting_module.predict(region_name, run_day, test_start_date, test_end_date)
        metric_results = self.evaluate_for_forecast(predictions, loss_functions)
        if output_filepath is not None:
            with open(output_filepath, 'w') as outfile:
                json.dump(metric_results, outfile)
        return metric_results


    def evaluate_for_forecast(self, forecast_df, loss_functions):
        metric_results = []
        for loss_function in loss_functions:
            # validate function is supported
            value = 0
            variables = loss_function["variables"]
            weights = loss_function["weights"]
            for i in range(len(variables)):
                value += weights[i] * (
                    getattr(metrics_util, loss_function["function_name"])(forecast_df["actual_" + variables[i]],
                                                                          forecast_df[variables[i]]))
            loss_function["value"] = value
            metric_results.append(loss_function)
        return metric_results

    @staticmethod
    def from_config(config_file_path):
        config = read_config_file(config_file_path)
        model_evaluator = ModelEvaluator(config["model_class"], config["model_parameters"])
        model_evaluator.evaluate(config["region_name"], config["run_day"],
                                 config["test_start_date"], config["test_end_date"],
                                 config["loss_functions"], config["output_filepath"])
