from configs.base_config import ForecastingModuleConfig
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
from utils.fetch_india_district_data import get_india_district_data_from_url, get_population_value
from entities.forecast_variables import ForecastVariable
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
                           forecast_end_date, model_parameters):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name)
        prediction_df = self.predict(region_type, region_name, region_metadata, observations, run_day, forecast_start_date,
                            forecast_end_date)
        dates = predictions_df['date']
        preddf = predictions_df.set_index('date')
        preddf = preddf.transpose().reset_index()
        preddf = preddf.rename(columns = {"index": "prediction_type", })
        error = float(model_parameters['MAPE'])/100
        for col in [ForecastVariable.active.name, ForecastVariable.hospitalized.name, ForecastVariable.icu.name,
            ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]:    
            series = preddf[preddf['prediction_type'] == col][dates]
            newSeries = series.multiply((1-error))
            newSeries['prediction_type'] = col+'_min'
            preddf = preddf.append(newSeries, ignore_index = True)
            newSeries = series.multiply((1+error))
            newSeries['prediction_type'] = col+'_max'
            preddf = preddf.append(newSeries, ignore_index = True)
            preddf.rename(columns = {col: col+'_mean'})
        preddf.insert(0, 'Province/State', region_name)
        preddf.insert(1, 'Country/Region', 'India')
        preddf.insert(2, 'Lat', 20)
        preddf.insert(3, 'Long', 70)
        return preddf.to_json()

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
                                                            config.forecast_end_date, config.model_parameters)
        if config.output_filepath is not None:
            predictions.to_csv(config.output_filepath, index=False)
        return predictions
