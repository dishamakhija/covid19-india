from entities.data_source import DataSource
from data_fetchers.data_fetcher_base import DataFetcherBase

import json
import pandas as pd
import numpy as np
from datetime import date
import urllib.request
from datetime import datetime
from io import StringIO  ## for Python 3
from functools import lru_cache

# raw input data from cocvid19india.org
raw_data_url = "https://api.covid19india.org/raw_data.json"
raw_data_urls = ["https://api.covid19india.org/raw_data1.json", "https://api.covid19india.org/raw_data2.json"]
post_april27_url = "https://api.covid19india.org/raw_data3.json"

INPUT_STATE_FIELD = "detectedstate"
INPUT_DISTRICT_FIELD = "detecteddistrict"
INPUT_CITY_FIELD = "detectedcity"
INPUT_DATE_ANNOUNCED_FIELD = "dateannounced"
INPUT_STATUS_CHANGE_DATE_FIELD = "statuschangedate"
NUM_CASES = 'numcases'

# column headers and static strings used in output CSV
REGION_TYPE = "region_type"
REGION_NAME = "region_name"
STATE = "state"
DISTRICT = "district"
CITY = "city"
OBSERVATION = "observation"
CONFIRMED = "confirmed"
DECEASED = "deceased"
RECOVERED = "recovered"
HOSPITALIZED = "hospitalized"
ACTIVE = "active"

@lru_cache(maxsize=3)
def load_observations_data():
    raw_data = []
    for url in raw_data_urls:
        raw_data.extend(get_raw_data_dict(url)["raw_data"])
        raw_data_post_april27 = get_raw_data_dict(post_april27_url)["raw_data"]
        df_list = []
        for region in [(INPUT_DISTRICT_FIELD, DISTRICT), (INPUT_CITY_FIELD, CITY), (INPUT_STATE_FIELD, STATE)]:
            for variable in [CONFIRMED, HOSPITALIZED, RECOVERED, DECEASED]:
                df = _get_covid_ts(raw_data, raw_data_post_april27, region[0], region[1], variable)
                df_list.append(df)
        merged_df = pd.concat(df_list)
        merged_df.reset_index(inplace=True)
        merged_df.to_csv("observations_latest.csv", index=False)
        return merged_df

def load_regional_metadata(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)

def get_raw_data_dict(input_url):
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict

def _get_covid_ts(stats, stats_post27_april, input_region_field, output_region_type_field, variable):
    if variable == CONFIRMED:
        df = pd.DataFrame \
            ([(i[input_region_field].lower().replace(',', ''), i[INPUT_DATE_ANNOUNCED_FIELD], 1) for i in stats])
        district_df_agg = pd.DataFrame(
            [(i[input_region_field].lower().replace(',', ''), i[INPUT_DATE_ANNOUNCED_FIELD], int(i[NUM_CASES]))
             for i in stats_post27_april
             if i['currentstatus'].lower() == HOSPITALIZED and NUM_CASES in i and i[NUM_CASES] != ''])
        df = df.append(district_df_agg)
    else:
        df = pd.DataFrame(
            [(i[input_region_field].lower().replace(',', ''), i[INPUT_STATUS_CHANGE_DATE_FIELD], 1) for i in stats if
             i['currentstatus'].lower() == variable])
        district_df_agg = pd.DataFrame(
            [(i[input_region_field].lower().replace(',', ''), i[INPUT_DATE_ANNOUNCED_FIELD], int(i[NUM_CASES]))
             for i in stats_post27_april
             if i['currentstatus'].lower() == variable and NUM_CASES in i and i[NUM_CASES] != ''])
        df = df.append(district_df_agg)
    df.columns = [REGION_NAME, 'date', 'counts']
    date_list = pd.date_range(start="2020-01-22", end=datetime.today()).strftime("%d/%m/%Y")
    df_pivot = pd.pivot_table(df, values='counts', index=[REGION_NAME], columns=['date']
                              , aggfunc={'counts': np.sum}, fill_value=0)
    df_pivot_dated = df_pivot.reindex(date_list, axis=1).fillna(0)
    df_final = df_pivot_dated.cumsum(axis=1)
    df_final.insert(0, REGION_TYPE, output_region_type_field)
    df_final.insert(1, OBSERVATION, variable)
    for date_val in date_list:
        datenew = datetime.strptime(date_val, "%d/%m/%Y")
        datenew = datetime.strftime(datenew, "%-m/%-d/%y")
        df_final = df_final.rename(columns={date_val: datenew})
    return df_final

class TrackerRaw(DataFetcherBase):

    def get_observations_for_region(self, region_type, region_name):
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        window_size, min_window_size = 3, 1
        date_col = 3    # Beginning of date column
        region_df.iloc[:,date_col:] = region_df.iloc[:,date_col:].rolling(window_size,axis=1, min_periods=min_window_size).mean()
        # if smooth:
        #     window_size, min_window_size = 3, 1
        #     date_col = 3    # Beginning of date column
        #     region_df.iloc[:,date_col:] = region_df.iloc[:,date_col:].rolling(window_size,axis=1, min_periods=min_window_size).mean()
        return region_df

    def get_regional_metadata(self, region_type, region_name, filepath="../data/regional_metadata.json"):
        metadata = load_regional_metadata(filepath)
        for params in metadata["regional_metadata"]:
            if params["region_type"] == region_type and params["region_name"] == region_name:
                return params["metadata"]