import os
import pdb
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from copy import deepcopy
from datetime import datetime, timedelta

from configs.base_config import TrainingModuleConfig
from configs.base_config import ModelEvaluatorConfig

from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule
from configs.base_config import ForecastingModuleConfig
from modules.forecasting_module import ForecastingModule
from modules.data_fetcher_module import DataFetcherModule


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
        if isinstance(parameters[param], dict):
            for key in parameters[param]:
                assert (not isinstance(parameters[param][key], dict))
                
                param_dict[interval + '_' + param + '_'+ key] = parameters[param][key]
        else:
            param_dict[interval + '_' + param] = parameters[param]
    return param_dict

def train_eval(configs,
               region, region_type, 
               train1_start_date, train1_end_date, 
               train2_start_date, train2_end_date, run_day,
               test_start_date, test_end_date, max_evals = 1000, 
               data_source = None, mlflow_log = True, name_prefix = None):
    """
        #TODO: Need to add hooks to consume data from appropriate source

        Run train and evalation for (basic) SEIR model.
    
    Arguments:
        region, region_type : Region info corresponding to the run
        train1_start_date, train1_end_date : Train1 durations
        train2_start_date, train2_end_date : Train2 durations
        test_start_date, test_end_date, run_day : Test durations
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
    
    # model parameters
    model_params = dict()
    model_params['region'] = region
    model_params['region_type'] = region_type
    model_params['model_type'] = "SEIR"
    

    train_config = deepcopy(configs['train'])
    train_config['region_name'] = region
    train_config['region_type'] = region_type
    train_config['train_start_date'] = train1_start_date
    train_config['train_end_date'] = train1_end_date
    train_config['search_parameters']['max_evals'] = max_evals
    
    if mlflow_log:
        train_config['output_filepath'] = 'train1_output.json'
    else:
        assert name_prefix is not None
        train_config['output_filepath'] = name_prefix + '_train1_output.json'

    train_module_config = TrainingModuleConfig.parse_obj(train_config)
    trainResults = TrainingModule.from_config(train_module_config)
    
    metrics['Train1MAPE'] = trainResults['train_metric_results'][0]['value']
    metrics['Train1RMLSE'] = trainResults['train_metric_results'][1]['value']
    metrics.update(parse_params(trainResults['best_params'], 'train1'))
    metrics.update(parse_params(trainResults['latent_params'], 'train1')) 
    
    test_config = deepcopy(configs['test'])
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
    
    metrics['TestMAPE'] = evalResults[0]['value']
    metrics['TestRMLSE'] = evalResults[1]['value']
    
    
    finalTrain_config = deepcopy(configs['train'])
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
    
    metrics['Train2MAPE'] = finalResults['train_metric_results'][0]['value']
    metrics['Train2RMLSE'] = finalResults['train_metric_results'][1]['value']
    metrics.update(parse_params(finalResults['best_params'], 'train2'))
    metrics.update(parse_params(finalResults['latent_params'], 'train2'))
        
    model_params['model_parameters'] = dict()
    model_params['model_parameters'].update(finalResults['best_params'])
    model_params['model_parameters'].update(finalResults['latent_params'])
    model_params['model_parameters']['MAPE'] = finalResults['train_metric_results'][0]['value']
    
    return params, metrics, model_params

def write_to_csv(dataframe, csv_fname):
    dataframe = dataframe.set_index('index')
    dataframe.to_csv(csv_fname)
    print("csv saved to {}".format(csv_fname))

def raw_data_to_csv(region, region_type):
    data = DataFetcherModule.get_observations_for_region(region_type, region)
    data = data.set_index('observation')
    data = data.transpose()
    data = data.reset_index()
    write_to_csv(data, csv_fname="results/raw_data_{}_{}.csv".format(region, region_type))

def plot(configs, model_params, run_day, forecast_start_date, forecast_end_date, 
         actual_start_date, plot_name = 'default.png'):
    evalConfig = ForecastingModuleConfig.parse_obj(configs['forecast'])
    evalConfig.region_name = model_params['region']
    evalConfig.region_type = model_params['region_type']
    evalConfig.model_parameters = model_params['model_parameters']

    evalConfig.run_day = run_day
    evalConfig.forecast_start_date = forecast_start_date
    evalConfig.forecast_end_date = forecast_end_date
    
    pdjson = ForecastingModule.from_config(evalConfig)
    pd_df = pd.read_json(pdjson)
    pd_df = pd_df.set_index('prediction_type')
    pd_df = pd_df.transpose()
    pd_df = pd_df.reset_index()
    pd_df = pd_df[5:]
    write_to_csv(pd_df, csv_fname="results/forecast_{}_{}_{}.csv".format(model_params['region'], 
                                                        forecast_start_date.replace('/','-'), 
                                                        forecast_end_date.replace('/','-')))

    actual = DataFetcherModule.get_observations_for_region(model_params['region_type'], model_params['region'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    actual = actual[start : ]
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title(model_params['region'])
    ax.plot(actual['index'], actual['confirmed'], color='green', label="observed")
    ax.plot(pd_df['index'], pd_df['confirmed_mean'], color='orange', label="predicted")
    plt.xticks(rotation=90)
    ax.set_ylim(ymin=0)
    ax.legend()
    fig.tight_layout()
    
    plt.savefig(plot_name)

def train_eval_plot(configs, region, region_type, 
                    train1_start_date, train1_end_date, 
                    train2_start_date, train2_end_date,
                    test_run_day, test_start_date, test_end_date, 
                    forecast_run_day, forecast_start_date, forecast_end_date,
                    max_evals = 1000, 
                    data_source = None, mlflow_log = True, name_prefix = None,
                    plot_name = 'results/default.png'):
    params, metrics, model_params = train_eval(configs, 
                                               region, region_type, 
                                               train1_start_date, train1_end_date, 
                                               train2_start_date, train2_end_date, 
                                               test_run_day, test_start_date, test_end_date, 
                                               max_evals, data_source, 
                                               mlflow_log, name_prefix)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv("./results/{}_{}_{}_{}_metrics.csv".format(region, region_type, forecast_start_date.replace('/','-'), forecast_end_date.replace('/','-')))

    model_params['model_parameters']['incubation_period'] = 5
    plot(configs, model_params, forecast_run_day, forecast_start_date, 
         forecast_end_date, actual_start_date=train1_start_date, plot_name=plot_name)

def dates_to_str(dates):
    str_dates = dict()
    for date in dates:
        str_dates[date] = dates[date].strftime("%-m/%-d/%y")
    return str_dates

def set_dates(train2_end_date, forecast_end_date):
    t = datetime.now().date()

    dates = defaultdict()

    dates["train1_start_date"] = (t - timedelta(14))
    dates["train1_end_date"] = (t - timedelta(8))

    if train2_end_date is None:
        dates["train2_start_date"] = (t - timedelta(7))
        dates["train2_end_date"] = (t - timedelta(1))
    else:
        dates["train2_end_date"] = datetime.strptime(train2_end_date, "%m/%d/%y")
        dates["train2_start_date"] = dates["train2_end_date"] - timedelta(6)

    dates["test_start_date"] = dates['train2_start_date']
    dates["test_end_date"] = dates['train2_end_date']
    dates["test_run_day"] = dates["test_start_date"] - timedelta(1)

    dates["forecast_start_date"] = dates["test_start_date"]
    dates["forecast_run_day"] = dates["forecast_start_date"] - timedelta(1)
    dates["forecast_end_date"] = datetime.strptime(forecast_end_date, "%m/%d/%y")

    dates = dates_to_str(dates)
    print(dates)

    return dates

def main(region=None, region_type=None, train2_end_date=None, forecast_end_date=None):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    raw_data_to_csv(region, region_type)
    print("forecasting for {}, {} till {}.".format(region, region_type, forecast_end_date))

    configs = dict()
    with open('train_config.json') as f_train, open('test_config.json') as f_test, open('forecast_config.json') as f_forecast:
        configs['train'] = json.load(f_train)
        configs['test'] = json.load(f_test)
        configs['forecast'] = json.load(f_forecast)

    dates = set_dates(train2_end_date, forecast_end_date)

    name_prefix = "./results/{}_{}".format(region, region_type)

    train_eval_plot(configs, region, region_type, 
                    dates['train1_start_date'], dates['train1_end_date'], 
                    dates['train2_start_date'], dates['train2_end_date'],
                    dates['test_run_day'], dates['test_start_date'], dates['test_end_date'], 
                    dates['forecast_run_day'], dates['forecast_start_date'], dates["forecast_end_date"],
                    max_evals = 1000, 
                    mlflow_log = False, name_prefix = name_prefix,
                    plot_name = './results/{}_{}.png'.format(region, forecast_end_date.replace('/', '-')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Covid forecast', allow_abbrev=False)
    parser.add_argument('--region', type=str, required=True, help = 'name of the region for which to get forecasts.')
    parser.add_argument('--region_type', type=str, required=True)
    parser.add_argument('--forecast_end_date', type=str, required=True, help = 'date till which forecast is required (mm-dd-yy)')
    parser.add_argument('--train2_end_date', type=str, default=None, help = '(optional) date till which train2 will run (mm-dd-yy)')
    (args, _) = parser.parse_known_args()

    main(region=args.region, region_type=args.region_type, forecast_end_date=args.forecast_end_date, train2_end_date=args.train2_end_date)
