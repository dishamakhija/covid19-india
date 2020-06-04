import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import math
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return math.sqrt(mse)

def rmse_delta(y_true, y_pred):
    y_true_delta = [y_true[i] - y_true[i-1] for i in range(1, len(y_true))]
    y_pred_delta = [y_pred[i] - y_pred[i-1] for i in range(1, len(y_pred))]
    mse = mean_squared_error(y_true_delta, y_pred_delta)
    return math.sqrt(mse)

def rmsle(y_true, y_pred):
    y_true_delta = [y_true[i] - y_true[i-1] for i in range(1, len(y_true))]
    y_pred_delta = [y_pred[i] - y_pred[i-1] for i in range(1, len(y_pred))]
    msle = mean_squared_log_error(y_true_delta, y_pred_delta)
    return math.sqrt(msle)

def rmsle_delta(y_true, y_pred):
    y_true_delta = [y_true[i] - y_true[i-1] for i in range(1, len(y_true))]
    y_pred_delta = [y_pred[i] - y_pred[i-1] for i in range(1, len(y_pred))]
    msle = mean_squared_log_error(y_true, y_pred)
    return math.sqrt(msle)

def mape(y_true, y_pred):
    mape_value = 0
    for i in range(len(y_pred)):
        if (not y_true[i] == 0):
            mape_value += np.abs((y_true[i] - y_pred[i] + 0.) / y_true[i])
    mape_value = (100 * mape_value) / len(y_pred)
    return mape_value

def mape_delta(y_true, y_pred):
    y_true_delta = [y_true[i] - y_true[i-1] for i in range(1, len(y_true))]
    y_pred_delta = [y_pred[i] - y_pred[i-1] for i in range(1, len(y_pred))]
    mape_value = 0
    for i in range(len(y_pred_delta)):
        if (not y_true_delta[i] == 0):
            mape_value += np.abs((y_true_delta[i] - y_pred_delta[i] + 0.) / y_true_delta[i])
    mape_value = (100 * mape_value) / len(y_pred_delta)
    return mape_value


def evaluate(y_true, y_pred):
    metrics = {}
    metrics["rmse"] = rmse(y_true, y_pred)
    metrics["rmsle"] = rmsle(y_true, y_pred)
    metrics["mape"] = mape(y_true, y_pred)
    metrics["mape_delta"] = mape_delta(y_true, y_pred)
    metrics["rmse_delta"] = rmse_delta(y_true, y_pred)
    metrics["rmsle_delta"] = rmsle_delta(y_true, y_pred)
    return metrics
