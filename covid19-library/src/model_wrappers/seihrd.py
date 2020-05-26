from datetime import timedelta, datetime
from functools import reduce

from entities.forecast_variables import ForecastVariable
from model_wrappers.base import ModelWrapperBase

import numpy as np
import pandas as pd

from seirsplus.models import *


class SEIHRD(ModelWrapperBase):

    def fit(self):
        pass

    def __init__(self, model_parameters: dict):
        self.model_parameters = model_parameters

    def supported_forecast_variables(self):
        return [ForecastVariable.confirmed, ForecastVariable.recovered, ForecastVariable.active]

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        search_space = kwargs.get("search_space", {})
        self._is_tuning = kwargs.get("is_tuning", False)
        self.model_parameters.update(search_space)
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(region_observations, region_metadata, run_day, n_days)
        date_list = list(pd.date_range(start=start_date, end=end_date).strftime("%-m/%-d/%y"))
        prediction_dataset = prediction_dataset[prediction_dataset.date.isin(date_list)]
        return prediction_dataset

    def is_black_box(self):
        return True

    def get_latent_params(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, end_date: str,
                          search_space: dict = {}):
        self.model_parameters.update(search_space)
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(region_observations, region_metadata, run_day, n_days)
        params = dict()
        params['latent_params'] = dict()
        params['latent_params']['LatentEbyCRatio'] = dict()
        params['latent_params']['LatentEbyCRatio'][run_day] = self.model_parameters.get("EbyCRatio")
        ed = prediction_dataset[prediction_dataset['date'] == end_date]
        params['latent_params']['LatentEbyCRatio'][end_date] = float(ed[ForecastVariable.exposed.name]) / float(
            ed[ForecastVariable.confirmed.name])
        params['latent_params']['LatentIbyCRatio'] = dict()
        params['latent_params']['LatentIbyCRatio'][run_day] = self.model_parameters.get("IbyCRatio")
        ed = prediction_dataset[prediction_dataset['date'] == end_date]
        params['latent_params']['LatentIbyCRatio'][end_date] = float(ed[ForecastVariable.active.name]) / float(
            ed[ForecastVariable.confirmed.name])
        return params

    def run(self, region_observations: pd.DataFrame, region_metadata, run_day: str, n_days: int):
        r0 = self.model_parameters['r0']
        init_sigma = 1. / self.model_parameters['incubation_period']
        init_beta = r0 * init_sigma
        init_gamma = 1. / self.model_parameters['infectious_period']
        confirmed_dataset = \
            region_observations[region_observations.observation == ForecastVariable.confirmed.name].iloc[0]
        recovered_dataset = \
            region_observations[region_observations.observation == ForecastVariable.recovered.name].iloc[0]
        deceased_dataset = \
            region_observations[region_observations.observation == ForecastVariable.deceased.name].iloc[0]
        hospitalized_dataset = \
            region_observations[region_observations.observation == ForecastVariable.hospitalized.name].iloc[0]       
        initN = region_metadata.get("population")
        
        if self._is_tuning:
            initE = confirmed_dataset[run_day] * self.model_parameters.get('EbyCRatio')
            initI = confirmed_dataset[run_day] * self.model_parameters.get('IbyCRatio')
            #initR = confirmed_dataset[run_day] * (1 - self.model_parameters.get('IbyCRatio'))
            initR = confirmed_dataset[run_day]
            initH = hospitalized_dataset[run_day]
            initF = recovered_dataset[run_day] + deceased_dataset[run_day]
        else:
            pick_day = run_day
            while (not pick_day in self.model_parameters.get('LatentEbyCRatio')):
                pick_day = (datetime.strptime(pick_day, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
            initE = confirmed_dataset[run_day] * self.model_parameters.get('LatentEbyCRatio').get(pick_day)
            initI = confirmed_dataset[run_day] * self.model_parameters.get('LatentIbyCRatio').get(pick_day)
            #initR = confirmed_dataset[run_day] * (1 - self.model_parameters.get('LatentIbyCRatio').get(pick_day))
            initR = confirmed_dataset[run_day]
            initH = hospitalized_dataset[run_day]
            initF = recovered_dataset[run_day] + deceased_dataset[run_day]
      
        estimator = SEIRSModel(beta=init_beta, sigma=init_sigma, gamma=init_gamma, initN=initN, initI=initI,
                               initE=initE, initR=initR)
        estimator.run(T=n_days, verbose=False)

        num_steps = len(estimator.numI)

        numF = np.zeros(num_steps)
        numF[0] = initF

        numH = np.zeros(num_steps)
        numH[0] = initH

        for i in range(1, num_steps):
            numF[i] = numF[i-1] + self.model_parameters["F_hospitalization"] * numH[i-1]
            numH[i] = estimator.numR[i] - numF[i]

        recovered_ts = self.alignTimeSeries(numF*(1 - self.model_parameters["F_fatalities"]), estimator.tseries, run_day, n_days, ForecastVariable.recovered.name)
        fatalities_ts = self.alignTimeSeries(numF*self.model_parameters["F_fatalities"], estimator.tseries, run_day, n_days, ForecastVariable.deceased.name)
        hospitalized_ts = self.alignTimeSeries(numH, estimator.tseries, run_day, n_days, ForecastVariable.hospitalized.name)
        icu_ts = self.alignTimeSeries(numH*self.model_parameters["F_icu"], estimator.tseries, run_day, n_days, ForecastVariable.icu.name)
        active_ts = self.alignTimeSeries(numH, estimator.tseries, run_day, n_days, ForecastVariable.active.name)
        confirmed_ts = self.alignTimeSeries(numH + numF, estimator.tseries, run_day, n_days, ForecastVariable.confirmed.name)
        exposed_ts = self.alignTimeSeries(estimator.numE, estimator.tseries, run_day, n_days, ForecastVariable.exposed.name)
        infected_ts = self.alignTimeSeries(estimator.numI, estimator.tseries, run_day, n_days, ForecastVariable.infected.name)
        final_ts = self.alignTimeSeries(numF, estimator.tseries, run_day, n_days, ForecastVariable.final.name)

        data_frames = [exposed_ts, icu_ts, recovered_ts, fatalities_ts, confirmed_ts, hospitalized_ts, active_ts, infected_ts, final_ts]
        result = reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'), data_frames)
        result = result.dropna()
        return result

    def alignTimeSeries(self, modelI, modelT, run_day, n_days, column_name=ForecastVariable.active.name):
        dates = [datetime.strptime(run_day, "%m/%d/%y") + timedelta(days=x) for x in range(n_days)]
        model_predictions = []
        count = 0
        day0 = dates[0]
        for date in dates:
            t = (date - day0).days
            while (modelT[count] <= t):
                count += 1
                if (count == len(modelT)):
                    print("Last prediction reached - Number of predictions less than required")
                    model_predictions.append(modelI[count - 1])
                    model_predictions_df = pd.DataFrame()
                    model_predictions_df['date'] = [date.strftime("%-m/%-d/%y") for date in dates]
                    model_predictions_df[column_name] = model_predictions
                    return model_predictions_df

            x0 = modelI[count] - (
                    ((modelI[count] - modelI[count - 1]) / (modelT[count] - modelT[count - 1])) * (modelT[count] - t))
            model_predictions.append(x0)
        model_predictions_df = pd.DataFrame()
        model_predictions_df['date'] = [date.strftime("%-m/%-d/%y") for date in dates]
        model_predictions_df[column_name] = model_predictions

        return model_predictions_df
