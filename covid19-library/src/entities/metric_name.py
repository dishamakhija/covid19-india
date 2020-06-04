import enum

@enum.unique
class MetricName(str, enum.Enum):
    rmse = "rmse"
    rmsle = "rmsle"
    mape = "mape"
    rmse_delta = "rmse_delta"
    rmsle_delta = "rmsle_delta"
    mape_delta = "mape_delta"
    
