from src.configs.base_config import ForecastingModuleConfig
from src.entities.model_class import ModelClass
from src.model_wrappers.model_factory import ModelFactory
from src.modules.data_fetcher_module import DataFetcherModule
from src.utils.config_util import read_config_file
from src.utils.fetch_india_district_data import get_india_district_data_from_url, get_population_value
import pandas as pd


class ForecastingModule(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def predict(self, region_type, region_name, region_metadata, region_observations, run_day, forecast_start_date,
                forecast_end_date):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        return predictions_df

    def predict_for_region(self, region_type, region_name, run_day, forecast_start_date,
                           forecast_end_date):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name)
        return self.predict(region_type, region_name, region_metadata, observations, run_day, forecast_start_date,
                            forecast_end_date)

    @staticmethod
    def from_config_file(config_file_path):
        config = read_config_file(config_file_path)
        forecasting_module_config = ForecastingModuleConfig.parse_obj(config)
        return ForecastingModule.from_config(forecasting_module_config)

    @staticmethod
    def from_config(config: ForecastingModuleConfig):
        forecasting_module = ForecastingModule(config.model_class, config.model_parameters)
        predictions = forecasting_module.predict_for_region(config.region_type, config.region_name,
                                                            config.run_day, config.forecast_start_date,
                                                            config.forecast_end_date)
        if config.output_filepath is not None:
            predictions.to_csv(config.output_filepath, index=False)
        return predictions
