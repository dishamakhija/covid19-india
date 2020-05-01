import pandas as pd

import abc

from entities.forecast_variables import ForecastVariable

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class ModelWrapperBase(ABC):
    @property
    @abc.abstractmethod
    def supported_forecast_variables(self):
        pass

    @abc.abstractmethod
    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def is_black_box(self):
        pass
