from datetime import timedelta, datetime
from functools import reduce

from entities.forecast_variables import ForecastVariable
from model_wrappers.base import ModelWrapperBase

import numpy as np
import pandas as pd

from seirsplus.models import *


class SEIR_gen(ModelWrapperBase):

    def fit(self):
        pass

    def __init__(self, model_parameters: dict):
        self.model_parameters = model_parameters
#         self.F_hospitalization = self.model_parameters.get("F_hospitalization", 0.04)
#         self.F_icu = self.model_parameters.get("F_icu", 0.05)
#         self.F_fatalities = self.model_parameters.get("F_fatalities", 0.08)


    def supported_forecast_variables(self):
        return [ForecastVariable.active, ForecastVariable.exposed, ForecastVariable.hospitalized, ForecastVariable.recovered, ForecastVariable.deceased]
    
    def input_variables(self):
        return [ForecastVariable.confirmed, ForecastVariable.recovered, ForecastVariable.deceased, ForecastVariable.hospitalized]

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, latent_variables: list, latent_on: ForecastVariable, **kwargs):
        search_space = kwargs.get("search_space", {})
        self._is_tuning = kwargs.get("is_tuning", False)
        self.model_parameters.update(search_space)
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(region_observations, region_metadata, run_day, n_days, latent_variables, latent_on)
        date_list = list(pd.date_range(start=start_date, end=end_date).strftime("%-m/%-d/%y"))
        prediction_dataset = prediction_dataset[prediction_dataset.date.isin(date_list)]
        return prediction_dataset

    def is_black_box(self):
        return True

    def get_latent_params(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, end_date: str,
                          search_space: dict = {}, latent_variables: list = [], 
                          latent_on: ForecastVariable = ForecastVariable.confirmed):
        self.model_parameters.update(search_space)
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(region_observations, region_metadata, run_day, n_days, latent_variables, latent_on)
        params = dict()
        params['latent_params'] = dict()
        ed = prediction_dataset[prediction_dataset['date'] == end_date]
#         rd = prediction_dataset[prediction_dataset['date'] == run_day]
        for latent_var in latent_variables:
            params['latent_params']["Latent_" + latent_var.name +"_ratio"] = dict()
            params['latent_params']["Latent_" + latent_var.name +"_ratio"][run_day] = self.model_parameters.get(latent_var.name+"_ratio")
            params['latent_params']["Latent_" + latent_var.name +"_ratio"][end_date] = float(ed[latent_var.name]) / float(
            ed[latent_on.name])
        return params

    def getOutsideToModelMap(self):
        # Should cover all the variables required by model as input
        d = dict()
        d['initI'] = ForecastVariable.active.name
        d['initE'] = ForecastVariable.exposed.name
        d['initH'] = ForecastVariable.hospitalized.name
        d['initR'] = ForecastVariable.recovered.name
        d['initD'] = ForecastVariable.deceased.name
        return d
    
    def getModelToOutside(self, estimator):
        # Should cover all the variables returned by supported_forecast_variables
        d = dict()
        d[ForecastVariable.exposed.name] = estimator.numE
        d[ForecastVariable.active.name] = estimator.numI
        d[ForecastVariable.hospitalized.name] = estimator.numH
        d[ForecastVariable.recovered.name] = estimator.numR
        d[ForecastVariable.deceased.name] = estimator.numD
        return d
        

    def get_model_with_params(self, region_metadata, region_observations, run_day,
                              latent_variables: list =[], 
                              latent_on: ForecastVariable = ForecastVariable.confirmed):
        r0 = self.model_parameters['r0']
        init_sigma = 1. / self.model_parameters['incubation_period']
        init_beta = r0 * init_sigma
        init_gamma = 1. / self.model_parameters['infectious_period']
        init_alpha = 1. / self.model_parameters['hospitalization_period']
        init_delta = 1. / self.model_parameters['deceased_ratio']
        initN = region_metadata.get("population")
        

        datasets = dict()
        initDict = dict()

        for var in self.input_variables():
            temp_dataset = region_observations[region_observations.observation == var.name].iloc[0]
            datasets[var.name] = temp_dataset

        if self._is_tuning:
            for var in latent_variables:
                initDict[var.name] = datasets[latent_on.name][run_day] * self.model_parameters.get(var.name + '_ratio')

        else:
            pick_day = run_day
            while (not pick_day in self.model_parameters.get("Latent_{}_ratio".format(latent_variables[0].name))):
                pick_day = (datetime.strptime(pick_day, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
            for var in latent_variables:
                initDict[var.name] = datasets[latent_on.name][run_day]* self.model_parameters.get('Latent_{}_ratio'.format(var.name)).get(pick_day)

         
        
        for var in self.input_variables():
            initDict[var.name] = datasets[var.name][run_day]

        oToM = self.getOutsideToModelMap()
        estimator = SEIRSModel(beta=init_beta, sigma=init_sigma, gamma=init_gamma, alpha=init_alpha, delta=init_delta,
                               initN=initN, initI=initDict[oToM['initI']], initE=initDict[oToM['initE']], 
                               initH=initDict[oToM['initH']], initR=initDict[oToM['initR']], initD=initDict[oToM['initD']])
        return estimator
            
        
    def run(self, region_observations: pd.DataFrame, region_metadata, run_day: str, n_days: int, 
            latent_variables: list, latent_on: ForecastVariable):

        
        estimator = self.get_model_with_params(region_metadata, region_observations, run_day, latent_variables, latent_on)
        
        estimator.run(T=n_days, verbose=False)
       
    
        mToO = self.getModelToOutside(estimator)
        data_frames = []
        for var in self.supported_forecast_variables():
            data_frames.append(self.alignTimeSeries(mToO[var.name], estimator.tseries, run_day, n_days, var.name))
            
            
        data_frames.append(self.alignTimeSeries([sum(x) for x in zip(estimator.numH, estimator.numR, estimator.numD)], estimator.tseries, run_day, n_days, ForecastVariable.confirmed.name))
        
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
    
    
#TODO
# 1. Infected vs Active for infectious bucket
