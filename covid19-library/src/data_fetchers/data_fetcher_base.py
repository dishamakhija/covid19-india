import abc

from entities.forecast_variables import ForecastVariable

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class DataFetcherBase(ABC):
    @property
    @abc.abstractmethod
    def get_observations_for_region(self, region_type: str, region_name:str):
        pass

    @abc.abstractmethod
    def get_regional_metadata(self, region_type: str, region_name: str, filepath: str):
        pass