from configs.base_config import ForecastingModuleConfig
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
from entities.forecast_variables import ForecastVariable
import pandas as pd

from utils.data_transformer_helper import convert_to_nhu_format


class ForecastingModule(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model_parameters = model_parameters
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def predict(self, region_type: str, region_name: str, region_metadata: dict, region_observations: pd.DataFrame,
                run_day: str, forecast_start_date: str,
                forecast_end_date: str):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        predictions_df = convert_to_nhu_format(predictions_df, region_type, region_name, self._model_parameters['MAPE'])
        return predictions_df.to_json()

    def predict_old_format(self, region_type: str, region_name: str, region_metadata: dict,
                           region_observations: pd.DataFrame,
                           run_day: str, forecast_start_date: str,
                           forecast_end_date: str):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        predictions_df = self.convert_to_old_required_format(run_day, predictions_df, region_type, region_name)
        return predictions_df.to_json()

    def convert_to_old_required_format(self, run_day, predictions_df, region_type, region_name):

        dates = predictions_df['date']
        preddf = predictions_df.set_index('date')
        columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name,
                   ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
        for col in columns:
            preddf = preddf.rename(columns={col: col + '_mean'})
        error = min(1, float(self._model_parameters['MAPE']) / 100)
        for col in columns:
            col_mean = col + '_mean'
            preddf[col + '_min'] = preddf[col_mean] * (1 - error)
            preddf[col + '_max'] = preddf[col_mean] * (1 + error)

        preddf.insert(0, 'run_day', run_day)
        preddf.insert(1, 'Region Type', region_type)
        preddf.insert(2, 'Region', region_name)
        preddf.insert(3, 'Model', self._model.__class__.__name__)
        preddf.insert(4, 'Error', "MAPE")
        preddf.insert(5, "Error Value", error * 100)

        return preddf

    def predict_for_region(self, region_type, region_name, run_day, forecast_start_date,
                           forecast_end_date):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name)
        return self.predict(region_type, region_name, region_metadata, observations, run_day,
                            forecast_start_date,
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
