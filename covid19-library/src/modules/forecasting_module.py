from typing import List

from src.entities.forecast_variables import ForecastVariable
from src.model_wrappers.base import ModelWrapperBase
from src.model_wrappers.model_factory import ModelFactory
from src.utils.config_util import read_config_file
from src.utils.fetch_india_district_data import get_india_district_data_from_url, get_population_value


class ForecastingModule(object):

    def __init__(self, model_class, model_parameters):
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def predict(self, region, run_day, forecast_start_date, forecast_end_date, filepath=None):
        confirmed_data = get_india_district_data_from_url(region, "confirmed")
        recovered_data = get_india_district_data_from_url(region, "Recovered")
        population = get_population_value(region)
        confirmed_data["population"] = population
        predictions = self._model.predict(confirmed_data, recovered_data, run_day, forecast_start_date, forecast_end_date)
        if filepath is not None:
            predictions.to_csv(filepath, index=False)
        return predictions

    @staticmethod
    def from_config(config_file_path):
        config = read_config_file(config_file_path)
        forecasting_module = ForecastingModule(config["model_class"], config["model_parameters"])
        forecasting_module.predict(config["region_name"], config["run_day"],
                                   config["forecast_start_date"], config["forecast_end_date"],
                                   config["output_filepath"])
