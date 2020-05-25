import json
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from datetime import datetime, timedelta

from configs.base_config import TrainingModuleConfig
from configs.base_config import ModelEvaluatorConfig

from modules.data_fetcher_module import DataFetcherModule
from modules.forecasting_module import ForecastingModule
from configs.base_config import ForecastingModuleConfig
from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule

import matplotlib.dates as mdates


def parse_params(parameters, interval='Train1'):
    """
        Flatten the params dictionary to enable logging
    of the parameters.

    Assumptions:
        There is a maximum of one level of nesting.
        Ensured using an assert statement for now.

    Sample_input:
        {
            'LatentEbyCRatio': {
                '4/7/20': 0.5648337712691847,
                '4/17/20': 1.1427545912005197
            },
            'LatentIbyCRatio': {
                '4/7/20': 0.9610881623714099,
                '4/17/20': 0.6742970940209254
            }
        }

    Output:
        {
            'Train1_LatentEbyCRatio_4/7/20': 0.5648337712691847,
            'Train1_LatentEbyCRatio_4/17/20': 1.1427545912005197,
            'Train1_LatentIbyCRatio_4/7/20': 0.9610881623714099,
            'Train1_LatentIbyCRatio_4/17/20': 0.6742970940209254
        }
    """
    param_dict = dict() # The flattened dictionary to return
    for param in parameters:
        if 'All' in param:
            continue
        if isinstance(parameters[param], dict):
            for key in parameters[param]:
                assert (not isinstance(parameters[param][key], dict))

                param_dict[interval + '_' + param + '_'+ key] = parameters[param][key]
        else:
            param_dict[interval + '_' + param] = parameters[param]
    return param_dict


def train_eval(region, region_type, 
               train1_start_date, train1_end_date, 
               train2_start_date, train2_end_date, run_day,
               test_start_date, test_end_date,
               default_train_config, default_test_config,
               max_evals = 1000, data_source = None, 
               mlflow_log = True, name_prefix = None):
    """
        #TODO: Need to add hooks to consume data from appropriate source

        Run train and evalation for (basic) SEIR model.
    
    Arguments:
        region, region_type : Region info corresponding to the run
        train1_start_date, train1_end_date : Train1 durations
        train2_start_date, train2_end_date : Train2 durations
        test_start_date, test_end_date, run_day : Test durations
        default_train_config : Default train config (loaded from train_config.json)
        default_test_config : Default test config (loaded from test_config.json)
        max_evals : number of search evaluations for SEIR (default: 1000)
        data_source : Data source for picking the region data
        mlflow_log : Experiment logged using MLFlow (default: True)
        name_prefix : In case of non-MLFlow experiment, string prefix to
                      enable easy indexing of experiments

    Note:
        date_format : %-m/%-d/%-y

    Returns: 
        params : Run parameters to be logged
        metrics : Metrics collected from the run 
    
    Output files saved : (name_prefix added in the case of non-MLflow experiments)
        Train1 : train1_output.json (name_prefix + '_train1_output.json')
        Train2 : train2_output.json (name_prefix + '_train2_output.json')
        Test   : test_output.json   (name_prefix + '_test_output.json')
    """
    
    # Save metrics and params for logging
    params = dict()
    metrics = dict()

    params['region'] = region
    params['region_type'] = region_type
    params['train1_start_date'] = train1_start_date
    params['train1_end_date'] = train1_end_date
    params['train2_start_date'] = train2_start_date
    params['train2_end_date'] = train2_end_date
    params['run_day'] = run_day
    params['test_start_date'] = test_start_date
    params['test_end_date'] = test_end_date
    params['data_source'] = data_source

    train_config = deepcopy(default_train_config)
    train_config['region_name'] = region
    train_config['region_type'] = region_type
    train_config['train_start_date'] = train1_start_date
    train_config['train_end_date'] = train1_end_date
    train_config['search_parameters']['max_evals'] = max_evals
    
    # model parameters
    model_params = dict()
    model_params['region'] = region
    model_params['region_type'] = region_type
    model_params['model_type'] = train_config['model_class']
    
    train1_model_params = deepcopy(model_params)
    train2_model_params = deepcopy(model_params)
       
    if mlflow_log:
        train_config['output_filepath'] = 'train1_output.json'
    else:
        assert name_prefix is not None
        train_config['output_filepath'] = name_prefix + '_train1_output.json'

    train_module_config = TrainingModuleConfig.parse_obj(train_config)
    trainResults = TrainingModule.from_config(train_module_config)
    
    train1MAPE = 0
    train1RMSLE = 0
    for metric in trainResults['train_metric_results']:
        if metric['metric_name'] == 'mape':
            train1MAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            train1RMSLE += metric['value']

    metrics['Train1RMLSE'] = train1RMSLE
    metrics['Train1MAPE'] = train1MAPE
    metrics['Train1All'] = trainResults['train_metric_results']
    metrics.update(parse_params(trainResults['best_params'], 'Train1'))
    metrics.update(parse_params(trainResults['latent_params'], 'Train1')) 
    train1_model_params['model_parameters'] = trainResults['model_parameters']
    
    test_config = deepcopy(default_test_config)
    test_config['region_name'] = region
    test_config['region_type'] = region_type
    test_config['test_start_date'] = test_start_date
    test_config['test_end_date'] = test_end_date
    test_config['run_day'] = run_day
    test_config['model_parameters'].update(trainResults['best_params'])    
    test_config['model_parameters'].update(trainResults['latent_params'])  
        
    if mlflow_log:
        test_config['output_filepath'] = 'test_output.json'
    else:
        test_config['output_filepath'] = name_prefix + '_test_output.json'

    test_module_config = ModelEvaluatorConfig.parse_obj(test_config) 
    evalResults = ModelEvaluator.from_config(test_module_config)
    
    testMAPE = 0
    testRMSLE = 0
    for metric in evalResults:
        if metric['metric_name'] == 'mape':
            testMAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            testRMSLE += metric['value']

    metrics['TestMAPE'] = testMAPE
    metrics['TestRMLSE'] = testRMSLE
    metrics['TestAll'] = evalResults
    
    finalTrain_config = deepcopy(default_train_config)
    finalTrain_config['region_name'] = region
    finalTrain_config['region_type'] = region_type
    finalTrain_config['train_start_date'] = train2_start_date
    finalTrain_config['train_end_date'] = train2_end_date
    finalTrain_config['search_parameters']['max_evals'] = max_evals
    
    if mlflow_log:
        finalTrain_config['output_filepath'] = 'train2_output.json'
    else:
        finalTrain_config['output_filepath'] = name_prefix + '_train2_output.json'

    finalTrain_module_config = TrainingModuleConfig.parse_obj(finalTrain_config)
    finalResults = TrainingModule.from_config(finalTrain_module_config)
    

    train2MAPE = 0
    train2RMSLE = 0
    for metric in finalResults['train_metric_results']:
        if metric['metric_name'] == 'mape':
            train2MAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            train2RMSLE += metric['value']

    metrics['Train2MAPE'] = train2MAPE
    metrics['Train2RMLSE'] = train2RMSLE
    metrics['Train2All'] = finalResults['train_metric_results']
    metrics.update(parse_params(finalResults['best_params'], 'Train2'))
    metrics.update(parse_params(finalResults['latent_params'], 'Train2'))
        
    model_params['model_parameters'] = finalResults['model_parameters']
    train2_model_params['model_parameters'] = finalResults['model_parameters']
    
    return params, metrics, train1_model_params, train2_model_params


def forecast(model_params, run_day, forecast_start_date, forecast_end_date,
             default_forecast_config):
    """
        Generate forecasts for a chosen interval using model parameters

    Arguments:
        model_params : Model parameters (dict)
        run_day : Date to initialize model paramters
        forecast_start_date, forecast_end_date : Forecast interval

    Returns:
        forecast_df : Dataframe containing forecasts
    """
    evalConfig = ForecastingModuleConfig.parse_obj(default_forecast_config)
    evalConfig.region_name = model_params['region']
    evalConfig.region_type = model_params['region_type']
    evalConfig.model_parameters = model_params['model_parameters']


    evalConfig.run_day = run_day
    evalConfig.forecast_start_date = forecast_start_date
    evalConfig.forecast_end_date = forecast_end_date

    forecast_json = ForecastingModule.from_config(evalConfig)
    forecast_df = pd.read_json(forecast_json)
    forecast_df = forecast_df.drop(columns=['Region Type', 'Region', 'Country', 'Lat', 'Long'])
    forecast_df = forecast_df.set_index('prediction_type')
    forecast_df = forecast_df.transpose().reset_index()
    return forecast_df


def get_observations_in_range(region_name, region_type, 
                              start_date, end_date,
                              obs_type = 'confirmed'):
    """
        Return a list of counts of obs_type cases
        from the region in the specified date range.
    """
    observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
    observations_df = observations[observations['observation'] == obs_type]
    
    start_date = datetime.strptime(start_date, '%m/%d/%y')
    end_date = datetime.strptime(end_date, '%m/%d/%y')
    delta = (end_date - start_date).days
    days = []
    for i in range(delta + 1):
        days.append((start_date + timedelta(days=i)).strftime('%-m/%-d/%-y'))
    
    # Fetch observations in the date range
    observations_df = observations_df[days]
    
    # Transpose the df to get the
    # observations_df.shape = (num_days, 1)
    observations_df = observations_df.reset_index(drop=True).transpose()
    
    # Rename the column to capture observation type
    # Note that the hardcoded 0 in the rename comes from the reset_index
    # from the previous step
    observations = observations_df[0].to_list()
    return observations


def plot(model_params, forecast_df, forecast_start_date, forecast_end_date, plot_name = 'default.png'):
    """
        Plot actual_confirmed cases vs forecasts.
        
        Assert that forecast_end_date is prior to the current date
        to ensure availability of actual_counts.
    """
    # Check for forecast_end_date being prior to current date
    end_date = datetime.strptime(forecast_end_date, '%m/%d/%y')
    assert end_date < datetime.now()    
    
    # Fetch actual counts from the DataFetcher module
    region_name = model_params['region']
    region_type = model_params['region_type']
    actual_observations = DataFetcherModule.get_observations_for_region(region_name, region_type)
    
    # Get relevant time-series of actual counts from actual_observations
    actual_observations = get_observations_in_range(region_name, region_type, 
                                                    forecast_start_date, 
                                                    forecast_end_date,
                                                    obs_type = 'confirmed')
    
    forecast_df['actual_confirmed'] = actual_observations
    
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(model_params['region'])
    ax.plot(forecast_df['index'], forecast_df['actual_confirmed'], color='blue', label="actual_confirmed")
    ax.plot(forecast_df['index'], forecast_df['confirmed_mean'], color='orange', label="predicted_confirmed")
    ax.set_ylim(ymin=0)
    ax.legend()

    plt.savefig(plot_name)

def train_eval_forecast(region, region_type,
                        train1_start_date, train1_end_date,
                        train2_start_date, train2_end_date,
                        test_run_day, test_start_date, test_end_date,
                        forecast_run_day, forecast_start_date, forecast_end_date,
                        default_train_config, default_test_config,
                        default_forecast_config, max_evals = 1000,
                        data_source = None, mlflow_log = True, name_prefix = None,
                        plot_actual_vs_predicted = False, plot_name = 'default.png'):
    """
        Run train, evaluation and generate forecasts as a dataframe.

        If plot_actual_vs_predicted is set to True,
        we first check if the forecast_end_date is prior to the current date
        so that we have actual_confirmed cases and then plot the predictions.
    """
    params, metrics, model_params = train_eval(region, region_type,
                                               train1_start_date, train1_end_date,
                                               train2_start_date, train2_end_date,
                                               test_run_day, test_start_date, test_end_date,
                                               default_train_config, default_test_config,
                                               max_evals, data_source,
                                               mlflow_log, name_prefix)
    model_params['model_parameters']['incubation_period'] = 5
    forecast_df = forecast(model_params, forecast_run_day, 
                           forecast_start_date, forecast_end_date, 
                           default_forecast_config)

    plot(model_params, forecast_df, forecast_start_date,
         forecast_end_date, plot_name=plot_name)

    return forecast_df, params, metrics, model_params


def plot_m1(train1_model_params, train1_run_day, train1_start_date, train1_end_date, 
            test_run_day, test_start_date, test_end_date, 
            rolling_average = False, uncertainty = False, 
            forecast_config = 'forecast_config.json',
            plot_config = 'plot_config.json', plot_name = 'default.png'):
    
    ## TODO: Log scale
    with open(plot_config) as fin:
        default_plot_config = json.load(fin)
    
    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average
    
    actual_start_date = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=7)).strftime("%-m/%-d/%y")    

    with open(forecast_config) as fin:
        default_forecast_config = json.load(fin)

    default_forecast_config['add_initial_observation'] = True
    
    # Get predictions
    pd_df_train = forecast(train1_model_params, train1_run_day, train1_start_date, train1_end_date, default_forecast_config)
    pd_df_test = forecast(train1_model_params, test_run_day, test_start_date, test_end_date, default_forecast_config)
    
    pd_df_train['index'] = pd.to_datetime(pd_df_train['index'])
    pd_df_test['index'] = pd.to_datetime(pd_df_test['index']) 
    pd_df_train = pd_df_train.sort_values(by=['index'])
    pd_df_test = pd_df_test.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(train1_model_params['region_type'], train1_model_params['region'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == test_end_date].tolist()[0]
    actual = actual[start : end+1]
    actual['index'] = pd.to_datetime(actual['index'])
    
    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    for variable in plot_variables:
        
        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'], 
                color = plot_colors[variable], label = plot_labels[variable]+': Observed')
        
        for pd_df in [pd_df_train, pd_df_test]:
        
            # Plot mean predictions
            if variable+'_mean' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_mean'], plot_markers['predicted']['mean'], 
                        color = plot_colors[variable], label = plot_labels[variable]+': Predicted')

            # Plot uncertainty in predictions
            if plot_config['uncertainty'] == True:

                if variable+'_min' in pd_df:
                    ax.plot(pd_df['index'], pd_df[variable+'_min'], plot_markers['predicted']['min'], 
                        color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Min)')

                if variable+'_max' in pd_df:
                    ax.plot(pd_df['index'], pd_df[variable+'_max'], plot_markers['predicted']['max'], 
                        color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Max)')

            # Plot rolling average
            if plot_config['rolling_average'] == True and variable+'_ra' in pd_df: 
                ax.plot(pd_df['index'], pd_df[variable+'_ra'], plot_markers['rolling_average'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (RA)')
    
    train_start = pd.to_datetime(train1_start_date)
    test_start = pd.to_datetime(test_start_date)
    
    line_height = plt.ylim()[1]
    ax.plot([train_start, train_start], [0,line_height], '--', color='brown', label='Train starts')
    ax.plot([test_start, test_start], [0,line_height], '--', color='black', label='Test starts')
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.title(train1_model_params['region'])
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.legend()
    plt.grid()
    
    plt.savefig(plot_name)

    
def plot_m2(train2_model_params, train_start_date, train_end_date, 
            test_run_day, test_start_date, test_end_date, 
            rolling_average = False, uncertainty = False, 
            forecast_config = 'forecast_config.json',
            plot_config = 'plot_config.json', plot_name = 'default.png'):
    
    ## TODO: Log scale
    with open(plot_config) as fplot, \
        open(forecast_config) as fcast:
        default_plot_config = json.load(fplot)
        default_forecast_config = json.load(fcast)
    
    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average
    
    default_forecast_config['add_initial_observation'] = True
    
    actual_start_date = (datetime.strptime(test_start_date, "%m/%d/%y") - timedelta(days=14)).strftime("%-m/%-d/%y")    
    
    # Get predictions
    pd_df_test = forecast(train2_model_params, test_run_day, test_start_date, test_end_date, default_forecast_config)
    
    pd_df_test['index'] = pd.to_datetime(pd_df_test['index']) 
    pd_df = pd_df_test.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(train2_model_params['region_type'], train2_model_params['region'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == test_end_date].tolist()[0]
    actual = actual[start : end+1]
    actual['index'] = pd.to_datetime(actual['index'])
    
    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    for variable in plot_variables:
        
        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'], 
                color = plot_colors[variable], label = plot_labels[variable]+': Observed')
        
        # Plot mean predictions
        if variable+'_mean' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable+'_mean'], plot_markers['predicted']['mean'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted')
        
        # Plot uncertainty in predictions
        if plot_config['uncertainty'] == True:
            
            if variable+'_min' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_min'], plot_markers['predicted']['min'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Min)')
                
            if variable+'_max' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_max'], plot_markers['predicted']['max'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Max)')
        
        # Plot rolling average
        if plot_config['rolling_average'] == True and variable+'_ra' in pd_df: 
            ax.plot(pd_df['index'], pd_df[variable+'_ra'], plot_markers['rolling_average'], 
                color = plot_colors[variable], label = plot_labels[variable]+': Predicted (RA)')
    
    test_start = pd.to_datetime(test_start_date)
    
    line_height = plt.ylim()[1]
    ax.plot([test_start, test_start], [0,line_height], '--', color='black', label='Test starts')
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.title(train2_model_params['region'])
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.legend()
    plt.grid()
    
    plt.savefig(plot_name)

    
def plot_m3(train2_model_params, train1_start_date, 
            forecast_start_date, forecast_length, 
            rolling_average = False, uncertainty = False,
            forecast_config = 'forecast_config.json',
            plot_config = 'plot_config.json', plot_name = 'default.png'):
    
    ## TODO: Log scale
    with open(plot_config) as fplot, \
        open(forecast_config) as fcast:
        default_plot_config = json.load(fplot)
        default_forecast_config = json.load(fcast)
    
    
    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average
    
    default_forecast_config['add_initial_observation'] = True
    
    actual_start_date = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=14)).strftime("%-m/%-d/%y")    
    forecast_run_day = (datetime.strptime(forecast_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
    forecast_end_date = (datetime.strptime(forecast_start_date, "%m/%d/%y") + timedelta(days=forecast_length)).strftime("%-m/%-d/%y")
    
    # Get predictions
    pd_df_forecast = forecast(train2_model_params, forecast_run_day, forecast_start_date, forecast_end_date, default_forecast_config)
    
    pd_df_forecast['index'] = pd.to_datetime(pd_df_forecast['index']) 
    pd_df = pd_df_forecast.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(train2_model_params['region_type'], train2_model_params['region'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == forecast_run_day].tolist()[0]
    actual = actual[start : end+1]
    actual['index'] = pd.to_datetime(actual['index'])
    
    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    for variable in plot_variables:
        
        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'], 
                color = plot_colors[variable], label = plot_labels[variable]+': Observed')
        
        # Plot mean predictions
        if variable+'_mean' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable+'_mean'], plot_markers['predicted']['mean'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted')
        
        # Plot uncertainty in predictions
        if plot_config['uncertainty'] == True:
            
            if variable+'_min' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_min'], plot_markers['predicted']['min'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Min)')
                
            if variable+'_max' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_max'], plot_markers['predicted']['max'], 
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Max)')
        
        # Plot rolling average
        if plot_config['rolling_average'] == True and variable+'_ra' in pd_df: 
            ax.plot(pd_df['index'], pd_df[variable+'_ra'], plot_markers['rolling_average'], 
                color = plot_colors[variable], label = plot_labels[variable]+': Predicted (RA)')
    
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.title(train2_model_params['region'])
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.legend()
    plt.grid()
    
    plt.savefig(plot_name)
