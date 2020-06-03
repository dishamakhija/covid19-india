import pandas as pd

from entities.data_source import DataSource
from data_fetchers.data_fetcher_factory import DataFetcherFactory

class DataFetcherModule(object):

    @staticmethod
    def get_observations_for_region(region_type, region_name, data_source = DataSource.tracker_district_daily, smooth = True):
        """
        region_type : Type of region (city, district, state)
        region_name : List of regions
        """
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        df = data_fetcher.get_observations_for_region(region_type, region_name)
        return df

    @staticmethod
    def get_regional_metadata(region_type, region_name, 
    data_source = DataSource.tracker_district_daily, filepath="../data/regional_metadata.json"):
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        metadata = data_fetcher.get_regional_metadata(region_type, region_name, filepath)
        return metadata


if __name__ == "__main__":
    pass
    