from typing import List

from configs.base_config import ScenarioForecastingModuleConfig, \
    ForecastTimeInterval
from entities.intervention_variable import InputType
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
import pandas as pd
from datetime import datetime, timedelta

from utils.data_transformer_helper import convert_to_initial_observations, convert_to_nhu_format


class ScenarioForecastingModule(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model_parameters = model_parameters
        self._model = ModelFactory.get_intervention_enabled_model(model_class, model_parameters)

    def predict(self, region_type: str, region_name: str, region_metadata: dict, region_observations: pd.DataFrame,
                run_day: str, start_date: str, input_type: InputType, time_intervals: List[ForecastTimeInterval]):
        """
                method calls predict method for model using region_metadata and region_observations
                @param region_observations: observations as fetched using data fetcher module
                @param region_metadata: region metadata providing demographic information
                @param region_type: region_type supported by data_fetcher module
                @param region_name: region_name supported by data_fetcher module
                @param run_day: date of initialization
                @param start_date: start_date
                @param input_type: input_type can be npi_list/param_override
                @param time_intervals: list of time_intervals with parameters
                @return: pd.DataFrame: predictions
        """
        run_day = run_day
        start_date = start_date
        predictions_list = []
        initial_observations = region_observations
        for time_interval in time_intervals:
            intervention_map = time_interval.get_interventions_map()
            predictions = self._model.predict_for_scenario(input_type, intervention_map, region_metadata,
                                                           initial_observations, run_day, start_date,
                                                           time_interval.end_date)
            predictions_list.append(predictions)
            ##setting run day, start date, initial_observations for next interval
            initial_observations = convert_to_initial_observations(predictions)
            run_day = time_interval.end_date
            start_date = (datetime.strptime(time_interval.end_date, "%m/%d/%y") + timedelta(days=1)).strftime(
                "%m/%d/%y")
        predictions_df = pd.concat(predictions_list, axis=0)
        predictions_df = convert_to_nhu_format(predictions_df, region_type, region_name, self._model_parameters['MAPE'])
        return predictions_df

    def predict_for_region(self, region_type: str, region_name: str, run_day: str, start_date: str,
                           input_type: InputType, time_intervals: List[ForecastTimeInterval]):
        """
        method downloads data using data fetcher module and then run predict on that dataset.
        @param region_type: region_type supported by data_fetcher module
        @param region_name: region_name supported by data_fetcher module
        @param run_day: date of initialization
        @param start_date: start_date
        @param input_type: input_type can be npi_list/param_override
        @param time_intervals: list of time_intervals with parameters
        @return: pd.DataFrame: predictions
        """
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name)
        return self.predict(region_type, region_name, region_metadata, observations, run_day,
                            start_date, input_type, time_intervals)

    @staticmethod
    def from_config_file(config_file_path: str):
        """
        method generates config class from config_file
        @param config_file_path: path to config_file
        @return: pd.DataFrame: predictions
        """
        config = read_config_file(config_file_path)
        forecasting_module_config = ScenarioForecastingModuleConfig.parse_obj(config)
        return ScenarioForecastingModule.from_config(forecasting_module_config)

    @staticmethod
    def from_config(config: ScenarioForecastingModuleConfig):
        """
        @param config: object of config class ScenarioForecastingModuleConfig
        @return: pd.DataFrame: predictions
        """
        forecasting_module = ScenarioForecastingModule(config.model_class, config.model_parameters)
        predictions = forecasting_module.predict_for_region(config.region_type, config.region_name,
                                                            config.run_day, config.start_date, config.input_type,
                                                            config.time_intervals)
        if config.output_filepath is not None:
            predictions.to_csv(config.output_filepath, index=False)
        return predictions
