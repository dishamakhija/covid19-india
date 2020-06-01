from entities.data_source import DataSource
from data_fetchers.data_fetcher_factory import DataFetcherFactory

import pandas as pd

REGION_TYPE = "region_type"
REGION_NAME = "region_name"
OBSERVATION = "observation"
CONFIRMED = "confirmed"
DECEASED = "deceased"
RECOVERED = "recovered"
HOSPITALIZED = "hospitalized"
ACTIVE = "active"

class DataFetcherModule(object):

    @staticmethod
    def get_observations_for_region_single(region_type, region_name, data_source = DataSource.tracker_district_daily, 
                                    smooth = True):
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        region_df = data_fetcher.get_observations_for_region_single(region_type, region_name)
        if smooth:
            window_size, min_window_size = 3, 1
            date_col = 3    # Beginning of date column
            region_df.iloc[:,date_col:] = region_df.iloc[:,date_col:].rolling(window_size,axis=1, min_periods=min_window_size).mean()
        return region_df


    @staticmethod
    def get_observations_for_region(region_type, region_name, combined_region_name = None, 
    data_source = DataSource.tracker_district_daily, smooth = True):
        """
        region_type : Type of region (city, district, state)
        region_name : List of regions
        """
        df_list = []
        if combined_region_name is None:
            combined_region_name = "-".join(region["region_name"] for region in region_name)
        for region in region_name:
            df_region = DataFetcherModule.get_observations_for_region_single(
                region_type, region, data_source = data_source, smooth = smooth)
            df_list.append(df_region)
        df = pd.concat(df_list)
        df = df.groupby([OBSERVATION]).sum().reset_index()
        df.insert(0, column=REGION_NAME, value=combined_region_name)
        df.insert(1, column=REGION_TYPE, value=region_type)
        return df

    @staticmethod
    def get_regional_metadata_single(region_type, region_name, 
    data_source = DataSource.tracker_district_daily, filepath="../data/regional_metadata.json"):
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        return data_fetcher.get_regional_metadata_single(region_type, region_name, filepath)

    @staticmethod
    def get_regional_metadata(region_type, region_name, 
    data_source = DataSource.tracker_district_daily, filepath="../data/regional_metadata.json"):
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        population = 0
        for region in region_name:
            regional_metadata = data_fetcher.get_regional_metadata_single(region_type, region, filepath)
            population += regional_metadata["population"]
        metadata = dict()
        metadata["population"] = population
        return metadata


if __name__ == "__main__":
    #print(DataFetcherModule.get_observations_for_region("district", "pune", "official_data"))
    print(DataFetcherModule.get_observations_for_region(
        "district",["mumbai", "thane"],"tracker_district_daily","../../data/regional_metadata.json"))
