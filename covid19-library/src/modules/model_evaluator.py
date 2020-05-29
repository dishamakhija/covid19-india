from typing import List

from configs.base_config import ModelEvaluatorConfig
from entities.forecast_variables import ForecastVariable
from entities.loss_function import LossFunction
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils import metrics_util
import json
import pandas as pd
from utils.config_util import read_config_file


class ModelEvaluator(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def evaluate(self, region_metadata, observations, run_day, test_start_date, test_end_date, loss_functions):
        predictions = self._model.predict(region_metadata, observations, run_day, test_start_date, test_end_date)
        return self.evaluate_for_forecast(observations, predictions, loss_functions)

    def evaluate_for_region(self, data_source, region_type, region_name, run_day, test_start_date, test_end_date, loss_functions):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, data_source)
        return self.evaluate(region_metadata, observations, run_day, test_start_date, test_end_date, loss_functions)

    @staticmethod
    def evaluate_for_forecast(observations, predictions_df, loss_functions: List[LossFunction]):
        metrics_results = []
        actual_df = ModelEvaluator.convert_dataframe(observations)
        actual_df = actual_df[actual_df.date.isin(predictions_df.date)]
        if(actual_df.shape[0] != predictions_df.shape[0]):
            raise Exception("Error in evaluation: number of rows don't match in predictions and actual dataframe" )
        for loss_function in loss_functions:
            value = 0
            for variable_weight in loss_function.variable_weights:
                variable_name = variable_weight.variable.name
                value += variable_weight.weight * (
                    getattr(metrics_util, loss_function.metric_name.name)(actual_df[variable_name].values,
                                                                          predictions_df[variable_name].values))
            loss_function.value = value
            metrics_results.append(loss_function.dict())
        return metrics_results

    @staticmethod
    def convert_dataframe( region_df):
        region_df  = region_df.reset_index()
        region_df.drop(["region_name", "region_type", "index"], axis=1, inplace=True)
        headers = region_df.transpose().reset_index().iloc[0]
        transposed_df = pd.DataFrame(region_df.transpose().reset_index().values[1:], columns=headers)
        transposed_df.rename({"observation": "date"}, axis='columns', inplace=True)
        return transposed_df

    @staticmethod
    def from_config(config: ModelEvaluatorConfig):
        model_evaluator = ModelEvaluator(config.model_class, config.model_parameters)
        metric_results = model_evaluator.evaluate_for_region(config.data_source, config.region_type, config.region_name, config.run_day,
                                                             config.test_start_date, config.test_end_date,
                                                             config.loss_functions)
        if config.output_filepath is not None:
            with open(config.output_filepath, 'w') as outfile:
                json.dump(metric_results, outfile, indent = 4)
        return metric_results

    @staticmethod
    def from_config_file(config_file_path: str):
        config = read_config_file(config_file_path)
        model_evaluator_config = ModelEvaluatorConfig.parse_obj(config)
        return ModelEvaluator.from_config(model_evaluator_config)
