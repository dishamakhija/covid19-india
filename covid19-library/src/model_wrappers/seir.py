from datetime import timedelta, datetime
from functools import reduce

from src.entities.forecast_variables import ForecastVariable
from src.model_wrappers.base import ModelWrapperBase
import pandas as pd
from seirsplus.models import *


class SEIR(ModelWrapperBase):

    def __init__(self, model_parameters: dict):
        self.model_parameters = model_parameters
        self.F_hospitalization = self.model_parameters.get("F_hospitalization", 0.26)
        self.F_icu = self.model_parameters.get("F_icu", 0.082)
        self.F_ventilation = self.model_parameters.get("F_ventilation", 0.05)
        self.F_fatalities = self.model_parameters.get("F_fatalities", 0.026)

    def supported_forecast_variables(self):
        return [ForecastVariable.total, ForecastVariable.recovered, ForecastVariable.infectious]

    def fit(self):
        raise Exception("This model doesn't support fit method")

    def transform_dataset(self, confirmed_data: pd.Series, recovered_data: pd.Series, run_day: str):
        confirmed_data['recovered_count'] = recovered_data[run_day]
        return confirmed_data

    def predict(self, confirmed_data: pd.Series, recovered_data: pd.Series, run_day: str, start_date: str,
                end_date: str):
        dataset = self.transform_dataset(confirmed_data, recovered_data, run_day)
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(dataset, run_day, n_days)
        date_list = list(pd.date_range(start=start_date, end=end_date).strftime("%-m/%-d/%y"))
        recovered_dataset = self.convert_dataframe(recovered_data, run_day, n_days,
                                                   "actual_" + ForecastVariable.recovered)
        prediction_dataset = prediction_dataset[prediction_dataset.date.isin(date_list)]
        prediction_dataset = pd.merge(prediction_dataset, recovered_dataset, left_on="date", right_on="date")
        return prediction_dataset

    def is_black_box(self):
        return True

    ##run_day => initialization_day
    def run(self, dataset: pd.Series, run_day: str, n_days: int):
        r0 = self.model_parameters['r0']
        init_sigma = 1. / self.model_parameters['incubation_period']
        init_beta = r0 * init_sigma
        init_gamma = 1. / self.model_parameters['infectious_period']
        initN = dataset['population']
        initI = dataset[run_day]
        initE = self.model_parameters.get('initE')
        if initE is None:
            initE = initI * self.model_parameters.get('EbyIRation')
        initR = dataset['recovered_count']
        estimator = SEIRSModel(beta=init_beta, sigma=init_sigma, gamma=init_gamma, initN=initN, initI=initI,
                               initE=initE, initR=initR)
        estimator.run(T=n_days, verbose=False)

        predicted_ts = self.alignTimeSeries(estimator.numI, estimator.tseries, run_day, n_days)
        recovered_ts = self.alignTimeSeries(estimator.numR * (1 - self.F_fatalities), estimator.tseries, run_day,
                                            n_days, ForecastVariable.recovered.name)
        fatalities_ts = self.alignTimeSeries(estimator.numR * self.F_fatalities, estimator.tseries, run_day, n_days,
                                             ForecastVariable.fatalities.name)
        region_row = self.convert_dataframe(dataset, run_day, n_days, "actual" + "_" + ForecastVariable.infectious.name)
        data_frames = [region_row, predicted_ts, recovered_ts, fatalities_ts]
        result = reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'), data_frames)
        result = result.dropna()
        return result

    def convert_dataframe(self, region_timeseries_df, run_day, n_days, column_name):
        dates_str = [(datetime.strptime(run_day, "%m/%d/%y") + timedelta(days=x)).strftime(format="%-m/%-d/%y").strip()
                     for x in range(n_days)]
        df = pd.DataFrame(region_timeseries_df[dates_str])
        df.columns = [column_name]
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        return df

    ##TODO: need to check this with harsh
    #### Since the model outputs a different format than the CSSEGISandData, we need to align them
    #### which means - finding the appropriate prediction for the model for a particular date
    ### modelI - Time Series Prediction output by model (model.numI)
    ### modelT - Time Series of the model (model.tseries)
    ### dates  - dates in the true file that we have from CSSEGISandData - datetime object
    def alignTimeSeries(self, modelI, modelT, run_day, n_days, column_name=ForecastVariable.infectious.name):
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
