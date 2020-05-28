from typing import List

from configs.base_config import ForecastingModuleConfig, ScenarioForecastingModuleConfig, ForecastScenario
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
from entities.forecast_variables import ForecastVariable
import pandas as pd
from datetime import datetime, timedelta

from utils.data_transformer_helper import _convert_to_initial_observations


class ScenarioForecastingModule(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model_parameters = model_parameters
        self._model = ModelFactory.get_intervention_enabled_model(model_class, model_parameters)

    def predict(self, region_type: str, region_name: str, region_metadata: dict, region_observations: pd.DataFrame,
                run_day: str, scenarios: List[ForecastScenario]):
        for scenario in scenarios:
            predictions_df = self.predict_for_scenario(region_type, region_name, region_metadata, region_observations,
                                                       run_day, scenario)
            return predictions_df ##TODO: fix this

    def predict_for_scenario(self, region_type: str, region_name: str, region_metadata: dict,
                             region_observations: pd.DataFrame, run_day: str,
                             scenario: ForecastScenario):
        run_day = run_day
        start_date = scenario.start_date
        predictions_list = []
        initial_observations = region_observations
        for time_interval in scenario.time_intervals:
            intervention_map = time_interval.get_interventions_map()
            predictions = self._model.predict_for_scenario(scenario.input_type, intervention_map, region_metadata,
                                                           initial_observations, run_day, start_date,
                                                           time_interval.end_date)
            predictions_list.append(predictions)
            ##setting run day, start date, initial_observations for next interval
            initial_observations = _convert_to_initial_observations(predictions)
            run_day = time_interval.end_date
            start_date = (datetime.strptime(time_interval.end_date, "%m/%d/%y") + timedelta(days=1)).strftime(
                "%m/%d/%y")
        predictions_df = pd.concat(predictions_list, axis=0)
        predictions_df = self.convert_to_required_format(predictions_df, region_type, region_name)
        return predictions_df

    def predict_old_format(self, region_type: str, region_name: str, region_metadata: dict,
                           region_observations: pd.DataFrame,
                           run_day: str, forecast_start_date: str,
                           forecast_end_date: str):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        predictions_df = self.convert_to_old_required_format(run_day, predictions_df, region_type, region_name)
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

    def convert_to_old_required_format(self, run_day, predictions_df, region_type, region_name):

        dates = predictions_df['date']
        preddf = predictions_df.set_index('date')
        columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name,
                   ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
        for col in columns:
            preddf = preddf.rename(columns={col: col + '_mean'})
        error = float(self._model_parameters['MAPE']) / 100
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

    def predict_for_region(self, region_type: str, region_name: str, run_day: str, scenarios: List[ForecastScenario]):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name)
        return self.predict(region_type, region_name, region_metadata, observations, run_day, scenarios)

    @staticmethod
    def from_config_file(config_file_path):
        config = read_config_file(config_file_path)
        forecasting_module_config = ScenarioForecastingModuleConfig.parse_obj(config)
        return ScenarioForecastingModule.from_config(forecasting_module_config)

    @staticmethod
    def from_config(config: ScenarioForecastingModuleConfig):
        forecasting_module = ScenarioForecastingModule(config.model_class, config.model_parameters)
        predictions = forecasting_module.predict_for_region(config.region_type, config.region_name,
                                                            config.run_day, config.scenarios)
        if config.output_filepath is not None:
            predictions.to_csv(config.output_filepath, index=False)
        return predictions
