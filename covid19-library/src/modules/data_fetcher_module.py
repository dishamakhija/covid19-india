# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import boto3
import uuid
import json
import pandas as pd
import numpy as np
from datetime import date
import urllib.request
from datetime import datetime
from io import StringIO  ## for Python 3

from functools import lru_cache

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# raw input data from cocvid19india.org
raw_data_url = "https://api.covid19india.org/raw_data.json"
raw_data_urls = ["https://api.covid19india.org/raw_data1.json", "https://api.covid19india.org/raw_data2.json"]
post_april27_url = "https://api.covid19india.org/raw_data3.json"
district_daily_url = 'https://api.covid19india.org/districts_daily.json'
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
def load_observations_data(data_source = 'tracker_district_daily'):
    raw_data = []
    if data_source == 'tracker_raw_data':
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
    elif data_source == 'direct_csv':
        pass
    elif data_source == 'tracker_district_daily':
        raw_data = get_raw_data_dict(district_daily_url)["districtsDaily"]
        dates = pd.date_range(start="2020-04-01",end=datetime.today()).strftime("%Y-%m-%d")
        columns =[REGION_NAME, OBSERVATION]
        columns.extend(dates)
        df = pd.DataFrame(columns=columns)
        for state_ut in raw_data:
            for district in raw_data[state_ut]:
                temp = pd.DataFrame(raw_data[state_ut][district])
                temp = temp.drop('notes', axis = 1) 
                temp = temp.set_index('date').transpose().reset_index().rename(columns={'index':OBSERVATION}).rename_axis(None)
                temp.insert(0, column=REGION_NAME, value=district.lower().replace(',', ''))
                df = pd.concat([df, temp], axis = 0, ignore_index = True, sort = False)
        df.insert(1, column = REGION_TYPE, value = DISTRICT)
        df = df.replace(ACTIVE, HOSPITALIZED)
        df = df.sort_values(by=[OBSERVATION])
        df = df.fillna(0)
        dates = pd.date_range(start="4/1/20",end=datetime.today()).strftime("%-m/%-d/%y")
        new_columns = [REGION_NAME, REGION_TYPE, OBSERVATION]
        new_columns.extend(dates)
        df = df.rename(columns=dict(zip(df.columns,new_columns)))
        return df
    else:
        pass


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


class DataFetcherModule(object):
    @staticmethod
    def get_observations_for_region(region_type, region_name, data_source = 'tracker_district_daily'):
        observations_df = load_observations_data(data_source = data_source)
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

    @staticmethod
    def get_regional_metadata(region_type, region_name, filepath="../data/regional_metadata.json"):
        metadata = load_regional_metadata(filepath)
        for params in metadata["regional_metadata"]:
            if params["region_type"] == region_type and params["region_name"] == region_name:
                return params["metadata"]


if __name__ == "__main__":
    DataFetcherModule.get_observations_for_region("district", "bengaluru")
