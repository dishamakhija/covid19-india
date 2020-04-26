import enum

@enum.unique
class MetricName(str, enum.Enum):
    rmse = "rmse"
    rmsle = "rmsle"
    mape = "mape"
