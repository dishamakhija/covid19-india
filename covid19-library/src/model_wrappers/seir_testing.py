import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime
import json

from model_wrappers.base import ModelWrapperBase


class SEIR_Testing(ModelWrapperBase):

    def __init__(self, R0=2.2, T_inf=2.9, T_inc=5.2, T_hosp=5, T_death=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14,
                 T_recov_mild=11, N=7e6, init_infected=1, intervention_day=100, intervention_amount=0.33, q=0,
                 testing_rate_for_exposed=0, positive_test_rate_for_exposed=1, testing_rate_for_infected=0,
                 positive_test_rate_for_infected=1, intervention_removal_day=45, intervention_removal_date='2020-05-18',
                 starting_date='2020-03-11', state_init_values=None, region_name='bengaluru', val_error=5):

        if type(starting_date) is str:
            starting_date = datetime.datetime.strptime(starting_date, '%Y-%m-%d')
        if type(intervention_removal_date) is str:
            intervention_removal_date = datetime.datetime.strptime(intervention_removal_date, '%Y-%m-%d')

        T_trans = T_inf/R0
        T_recov_mild = (14 - T_inf)
        T_recov_severe = (31.5 - T_inf)

        P_mild = 1 - P_severe - P_fatal

        # define testing related parameters
        T_inf_detected = T_inf
        T_trans_detected = T_trans
        T_inc_detected = T_inc

        P_mild_detected = P_mild
        P_severe_detected = P_severe
        P_fatal_detected = P_fatal

        vanilla_params = {

            'R0': R0,

            'T_trans': T_trans,
            'T_inc': T_inc,
            'T_inf': T_inf,

            'T_recov_mild': T_recov_mild,
            'T_recov_severe': T_recov_severe,
            'T_hosp': T_hosp,
            'T_death': T_death,

            'P_mild': P_mild,
            'P_severe': P_severe,
            'P_fatal': P_fatal,
            'intervention_day': intervention_day,
            'intervention_removal_day': (intervention_removal_date - starting_date).days,
            'intervention_amount': intervention_amount,
            'starting_date': starting_date,
            'N': N,
            'region_name' : region_name,
            'val_error' : val_error
        }

        testing_params = {
            'T_trans': T_trans_detected,
            'T_inc': T_inc_detected,
            'T_inf': T_inf_detected,

            'P_mild': P_mild_detected,
            'P_severe': P_severe_detected,
            'P_fatal': P_fatal_detected,

            'q': q,
            'testing_rate_for_exposed': testing_rate_for_exposed,
            'positive_test_rate_for_exposed': positive_test_rate_for_exposed,
            'testing_rate_for_infected': testing_rate_for_infected,
            'positive_test_rate_for_infected': positive_test_rate_for_infected
        }

        if state_init_values == None:
            # S, E, D_E, D_I, I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D
            state_init_values = OrderedDict()
            state_init_values['S'] = (N - init_infected)/N
            state_init_values['E'] = 0
            state_init_values['I'] = init_infected/N
            state_init_values['D_E'] = 0
            state_init_values['D_I'] = 0
            state_init_values['R_mild'] = 0
            state_init_values['R_severe_home'] = 0
            state_init_values['R_severe_hosp'] = 0
            state_init_values['R_fatal'] = 0
            state_init_values['C'] = 0
            state_init_values['D'] = 0

        for param_dict_name in ['vanilla_params', 'testing_params', 'state_init_values']:
            setattr(self, param_dict_name, eval(param_dict_name))

    def get_derivative(self, t, y):

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D = y

        # Init time parameters and probabilities
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])

        for key in self.testing_params:
            suffix = '_D' if key in self.vanilla_params else ''
            setattr(self, key + suffix, self.testing_params[key])

        # Modelling the intervention
        if t >= self.intervention_day:
            self.R0 = self.intervention_amount * self.R0
            self.T_trans = self.T_inf/self.R0

        # Modelling the intervention
        if t >= self.intervention_removal_day:
            self.R0 = self.R0 / self.intervention_amount
            self.T_trans = self.T_inf/self.R0

        # Init derivative vector
        dydt = np.zeros(y.shape)
        
        theta_E = self.testing_rate_for_exposed
        psi_E = self.positive_test_rate_for_exposed
        theta_I = self.testing_rate_for_infected
        psi_I = self.positive_test_rate_for_infected

        # Write differential equations
        dydt[0] = - I * S / (self.T_trans) - (self.q / self.T_trans_D) * (S * D_I)
        dydt[1] = I * S / (self.T_trans) + (self.q / self.T_trans_D) * (S * D_I) - (E/ self.T_inc) - (theta_E * psi_E * E)
        dydt[2] = E / self.T_inc - I / self.T_inf - (theta_I * psi_I * I)
        dydt[3] = (theta_E * psi_E * E) - (1 / self.T_inc_D) * D_E
        dydt[4] = (theta_I * psi_I * I) + (1 / self.T_inc_D) * D_E - (1 / self.T_inf_D) * D_I
        dydt[5] = (1/self.T_inf)*(self.P_mild*I) + (1/self.T_inf_D)*(self.P_mild_D*D_I) - R_mild/self.T_recov_mild
        dydt[6] = (1/self.T_inf)*(self.P_severe*I) + (1/self.T_inf_D)*(self.P_severe_D*D_I) - R_severe_home/self.T_hosp 
        dydt[7] = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe
        dydt[8] = (1/self.T_inf)*(self.P_fatal*I) + (1/self.T_inf_D)*(self.P_fatal_D*D_I) - R_fatal/self.T_death
        dydt[9] = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe
        dydt[10] = R_fatal/self.T_death

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1, method='Radau'):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)
        
        state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]

        sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                        state_init_values_arr, method=method, t_eval=time_steps)

        return sol

    def return_predictions(self, sol):
        states_time_matrix = (sol.y*self.vanilla_params['N']).astype('int')
        dataframe_dict = {}
        for i, key in enumerate(self.state_init_values.keys()):
            dataframe_dict[key] = states_time_matrix[i]
        
        df_prediction = pd.DataFrame.from_dict(dataframe_dict)
        df_prediction['date'] = pd.date_range(self.starting_date, self.starting_date + datetime.timedelta(days=df_prediction.shape[0] - 1))
        columns = list(df_prediction.columns)
        columns.remove('date')
        df_prediction = df_prediction[['date'] + columns]

        df_prediction['hospitalisations'] = df_prediction['R_severe_home'] + df_prediction['R_severe_hosp'] + df_prediction['R_fatal']
        df_prediction['recoveries'] = df_prediction['C']
        df_prediction['fatalities'] = df_prediction['D']
        df_prediction['infectious_unknown'] = df_prediction['I'] + df_prediction['D_I']
        df_prediction['total_infected'] = df_prediction['hospitalisations'] + df_prediction['recoveries'] + df_prediction['fatalities']
        return df_prediction

    def supported_forecast_variables(self):
        pass

    def modify_dataframe(self, df):
        columns = df['observation']
        df = df.iloc[:, 2:].T
        df.columns = columns.to_numpy()
        df = df.iloc[1:, :4]
        df.reset_index(inplace=True)
        df.columns = ['date', 'total_infected', 'hospitalized', 'recovered', 'deceased']
        df = df.loc[:, ['date', 'total_infected']]
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['total_infected'] != 0].reset_index(drop=True)
        last_change_index = np.argwhere(df['total_infected'].diff().to_numpy() > 0)[-1][0]
        df = df.loc[:last_change_index, :]
        return df

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, 
                      forecast_start_date: str, forecast_end_date: str, **kwargs):
        region_observations = self.modify_dataframe(region_observations)
        run_day = datetime.datetime.strptime(run_day, '%m/%d/%y')
        forecast_start_date = datetime.datetime.strptime(forecast_start_date, '%m/%d/%y')
        forecast_end_date = datetime.datetime.strptime(forecast_end_date, '%m/%d/%y')
        total_days = (forecast_end_date - self.vanilla_params['starting_date']).days
        sol = self.solve_ode(total_no_of_days=total_days, time_step=1, method='Radau')
        df_prediction = self.return_predictions(sol)
        df_prediction = df_prediction[df_prediction['date'] >= forecast_start_date]

        # Get min and max bounds using the calculated val error
        columns = ['predictionDate', 'active_mean', 'active_min', 'active_max', 'hospitalized_mean', 
                   'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 'icu_max', 'deceased_mean', 
                   'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 'recovered_max', 'total_mean', 
                   'total_min', 'total_max']

        df_output = pd.DataFrame(columns = columns)
        
        city = self.vanilla_params['region_name']
        error = self.vanilla_params['val_error']
        df_output['predictionDate'] = df_prediction['date']    
        pred_hospitalisations = df_prediction['hospitalisations']
        df_output['active_mean'] = pred_hospitalisations
        df_output['active_min'] = (1 - 0.01*error)*pred_hospitalisations
        df_output['active_max'] = (1 + 0.01*error)*pred_hospitalisations
        
        df_output['hospitalized_mean'] = df_prediction['hospitalisations']
        df_output['hospitalized_min'] = (1 - 0.01*error)*pred_hospitalisations
        df_output['hospitalized_max'] = (1 + 0.01*error)*pred_hospitalisations
        
        df_output['icu_mean'] = 0.02*pred_hospitalisations
        df_output['icu_min'] = (1 - 0.01*error)*0.02*pred_hospitalisations
        df_output['icu_max'] = (1 + 0.01*error)*0.02*pred_hospitalisations
        
        pred_recoveries = df_prediction['recoveries']
        df_output['recovered_mean'] = pred_recoveries
        df_output['recovered_min'] = (1 - 0.01*error)*pred_recoveries
        df_output['recovered_max'] = (1 + 0.01*error)*pred_recoveries
        
        pred_fatalities = df_prediction['fatalities']
        df_output['deceased_mean'] = pred_fatalities
        df_output['deceased_min'] = (1 - 0.01*error)*pred_fatalities
        df_output['deceased_max'] = (1 + 0.01*error)*pred_fatalities
        
        pred_total_cases = df_prediction['total_infected']
        df_output['total_mean'] = pred_total_cases
        df_output['total_min'] = (1 - 0.01*error)*pred_total_cases
        df_output['total_max'] = (1 + 0.01*error)*pred_total_cases
        
        # Convert to JSON format
        df_output.reset_index(inplace=True, drop=True)
        df_output.set_index('predictionDate', inplace=True)
        df_output = df_output.T
        df_output.index.rename('prediction_type', inplace=True)
        df_output.columns.rename('', inplace=True)
        df_output.columns = [datetime.datetime.strftime(x.date(), '%d/%m/%Y') for x in df_output.columns]
        df_output.reset_index(inplace=True)
        df_output.insert(loc=0, column='Province/State', value=[city.title()]*len(df_output.index))
        df_output.insert(loc=1, column='Country/Region', value=['India']*len(df_output.index))
        return df_output

    def fit(self):
        pass

    def is_black_box(self):
        return True


"""
Lines following this will be removed upon completion of this model class.
This is just for easy debugging.
"""
if __name__ == '__main__':
    from modules.data_fetcher_module import DataFetcherModule

    region_type = 'district'
    region_name = 'bengaluru'
    run_day = '3/22/20'
    forecast_start_date = '4/1/20'
    forecast_end_date = '4/20/20'
    output_save_path = '../seir_testing_evaluation_output.json'

    region_observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
    regional_metadata_path = '../../../covid19-library/data/regional_metadata.json'
    region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, filepath=regional_metadata_path)

    json_file_path = '../model_params_seir_t.json'
    with open(json_file_path, 'r') as j:
        master_model_params_dict = json.loads(j.read())
    for region_dict in master_model_params_dict['regional_model_params']:
        if region_dict['region_name'] == region_name:
            break
    
    model_params = region_dict['model_parameters']
    model_params['N'] = region_metadata['population']
    model_params['val_error'] = region_dict['val_error']
    model = SEIR_Testing(**model_params)
    df_output = model.predict(region_metadata, region_observations, run_day, forecast_start_date, forecast_end_date)
    import pdb; pdb.set_trace()
    df_output.to_json(output_save_path, orient='records')

    print('abc')
