import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import math
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_rmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(rmse)
    return rmse


def calculate_rmsle(y_true, y_pred):
    rmsle = mean_squared_log_error(y_true, y_pred)
    rmsle = math.sqrt(rmsle)
    return rmsle


def calculate_mape(y_true, y_pred):
    mape = 0
    for i in range(len(y_pred)):
        if (not y_true[i] == 0):
            mape += np.abs((y_true[i] - y_pred[i] + 0.) / y_true[i])
    mape = (100 * mape) / len(y_pred)
    return mape


def evaluate(y_true, y_pred):
    metrics = {}
    metrics["rmse"] = calculate_rmse(y_true, y_pred)
    metrics["rmsle"] = calculate_rmsle(y_true, y_pred)
    metrics["mape"] = calculate_mape(y_true, y_pred)
    return metrics

