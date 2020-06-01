from entities.data_source import DataSource
from data_fetchers.data_fetcher_base import DataFetcherBase

import json
import pandas as pd
import numpy as np
from datetime import date
import urllib.request
from datetime import datetime

from functools import lru_cache

# raw input data from cocvid19india.org
district_daily_url = 'https://api.covid19india.org/districts_daily.json'

# column headers and static strings used in output CSV
REGION_TYPE = "region_type"
REGION_NAME = "region_name"
DISTRICT = "district"
OBSERVATION = "observation"
CONFIRMED = "confirmed"
DECEASED = "deceased"
RECOVERED = "recovered"
HOSPITALIZED = "hospitalized"
ACTIVE = "active"

@lru_cache(maxsize=3)
def load_observations_data():
    raw_data = get_raw_data_dict(district_daily_url)["districtsDaily"]
    dates = pd.date_range(start="2020-04-01",end = datetime.today()).strftime("%Y-%m-%d")
    columns = [REGION_NAME, OBSERVATION]
    columns.extend(dates)
    df = pd.DataFrame(columns = columns)
    for state_ut in raw_data: 
        for district in raw_data[state_ut]:
            temp = pd.DataFrame(raw_data[state_ut][district])
            temp = temp.drop('notes', axis = 1) 
            temp = temp.set_index('date').transpose().reset_index().rename(columns={'index':OBSERVATION}).rename_axis(None)
            temp.insert(0, column = REGION_NAME, value = district.lower().replace(',', ''))
            df = pd.concat([df, temp], axis = 0, ignore_index = True, sort = False)
    df.insert(1, column = REGION_TYPE, value = DISTRICT)
    df = df.replace(ACTIVE, HOSPITALIZED)
    df = df.sort_values(by = [OBSERVATION])
    df = df.fillna(0)
    dates = pd.date_range(start="4/1/20",end = datetime.today()).strftime("%-m/%-d/%y")
    new_columns = [REGION_NAME, REGION_TYPE, OBSERVATION]
    new_columns.extend(dates)
    df = df.rename(columns = dict(zip(df.columns,new_columns)))
    return df


def load_regional_metadata(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)


def get_raw_data_dict(input_url):
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict

class TrackerDistrictDaily(DataFetcherBase):

    def get_observations_for_region_single(self, region_type, region_name):
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

    def get_regional_metadata_single(self, region_type, region_name, filepath):
        metadata = load_regional_metadata(filepath)
        for params in metadata["regional_metadata"]:
            if params["region_type"] == region_type and params["region_name"] == region_name:
                return params["metadata"]
