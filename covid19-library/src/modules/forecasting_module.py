from configs.base_config import ForecastingModuleConfig
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
from entities.forecast_variables import ForecastVariable
import pandas as pd


class ForecastingModule(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model_parameters = model_parameters
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def predict(self, region_type: str, region_name: str, region_metadata: dict, region_observations: pd.DataFrame,
                run_day: str, forecast_start_date: str,
                forecast_end_date: str):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        predictions_df = self.convert_to_required_format(predictions_df, region_type, region_name)
        return predictions_df.to_json()

    def convert_to_required_format(self, predictions_df, region_type, region_name):
        dates = predictions_df['date']
        preddf = predictions_df.set_index('date')
        columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name, ForecastVariable.icu.name,
                   ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
        for col in columns:
            preddf = preddf.rename(columns={col: col + '_mean'})
        preddf = preddf.transpose().reset_index()
        preddf = preddf.rename(columns={"index": "prediction_type", })
        error = float(self._model_parameters['MAPE']) / 100
        for col in columns:
            col_mean = col + '_mean'
            series = preddf[preddf['prediction_type'] == col_mean][dates]
            newSeries = series.multiply((1 - error))
            newSeries['prediction_type'] = col + '_min'
            preddf = preddf.append(newSeries, ignore_index=True)
            newSeries = series.multiply((1 + error))
            newSeries['prediction_type'] = col + '_max'
            preddf = preddf.append(newSeries, ignore_index=True)
            preddf = preddf.rename(columns={col: col + '_mean'})
        preddf.insert(0, 'Region Type', region_type)
        preddf.insert(1, 'Region', region_name)
        preddf.insert(2, 'Country', 'India')
        preddf.insert(3, 'Lat', 20)
        preddf.insert(4, 'Long', 70)
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
