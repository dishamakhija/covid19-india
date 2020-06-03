import abc
import pandas as pd

from typing import List

from entities.forecast_variables import ForecastVariable
from data_fetchers.data_fetcher_utils import load_regional_metadata

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class DataFetcherBase(ABC):
    @property
    @abc.abstractmethod
    def get_observations_for_single_region(self, region_type: str, region_name:str):
        pass

    def get_single_regional_metadata(self, region_type: str, region_name: str, filepath: str):
        metadata = load_regional_metadata(filepath)
        for params in metadata["regional_metadata"]:
            if params["region_type"] == region_type and params["region_name"] == region_name:
                return params["metadata"]


    def get_observations_for_region(self, region_type: str, region_name: List[str], smooth: bool = True):
        """
        region_type : Type of region (city, district, state)
        region_name : List of regions
        """
        df_list = []

        for region in region_name:
            df_region = self.get_observations_for_single_region(region_type, region)
            df_list.append(df_region)
        df = pd.concat(df_list, sort = False)

        combined_region_name = " ".join(region_name)

        if len(region_name) > 1:
            cols = [col for col in list(df) if col not in {"region_name","region_type","observation"}]
            df = df.groupby(["observation"])[cols].sum().reset_index()
            df.insert(0, column="region_name", value=combined_region_name)
            df.insert(1, column="region_type", value=region_type)

        if smooth:
            window_size, min_window_size = 3, 1
            date_col = 3    # Beginning of date column
            df.iloc[:,date_col:] = df.iloc[:,date_col:].rolling(window_size,axis=1, min_periods=min_window_size).mean()

        return df


    def get_regional_metadata(self, region_type: str, region_name: str, filepath: str):
        """
        region_type : Type of region (city, district, state)
        region_name : List of regions
        filepath : Path to file containing regional metadata

        Note:
            An assumption is made that metadata is only additive.
        """
        metadata = dict()
        for region in region_name:
            regional_metadata = self.get_single_regional_metadata(region_type, region, filepath)
            for key in regional_metadata.keys():
                if key not in metadata:
                    metadata[key] = 0
                metadata[key] += regional_metadata[key]
        return metadata
        