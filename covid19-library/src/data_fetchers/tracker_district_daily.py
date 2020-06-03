import json
import pandas as pd
import numpy as np
from datetime import datetime

from functools import lru_cache

from entities.data_source import DataSource
from data_fetchers.data_fetcher_base import DataFetcherBase
from data_fetchers.data_fetcher_utils import get_raw_data_dict

# raw input data from cocvid19india.org
district_daily_url = 'https://api.covid19india.org/districts_daily.json'

@lru_cache(maxsize=3)
def load_observations_data():
    raw_data = get_raw_data_dict(district_daily_url)["districtsDaily"]
    dates = pd.date_range(start="2020-04-01",end = datetime.today()).strftime("%Y-%m-%d")
    columns = ["region_name", "observation"]
    columns.extend(dates)
    df = pd.DataFrame(columns = columns)
    for state_ut in raw_data: 
        for district in raw_data[state_ut]:
            temp = pd.DataFrame(raw_data[state_ut][district])
            temp = temp.drop('notes', axis = 1) 
            temp = temp.set_index('date').transpose().reset_index().rename(columns={'index':"observation"}).rename_axis(None)
            temp.insert(0, column = "region_name", value = district.lower().replace(',', ''))
            df = pd.concat([df, temp], axis = 0, ignore_index = True, sort = False)
    df.insert(1, column = "region_type", value = "district")
    df = df.replace("active", "hospitalized")
    df = df.sort_values(by = ["observation"])
    df = df.fillna(0)
    dates = pd.date_range(start="4/1/20",end = datetime.today()).strftime("%-m/%-d/%y")
    new_columns = ["region_name", "region_type", "observation"]
    new_columns.extend(dates)
    df = df.rename(columns = dict(zip(df.columns,new_columns)))
    return df

class TrackerDistrictDaily(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name):
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

