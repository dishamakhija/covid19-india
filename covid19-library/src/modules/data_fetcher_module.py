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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# raw input data from cocvid19india.org
raw_data_url = "https://api.covid19india.org/raw_data.json"
INPUT_STATE_FIELD = "detectedstate"
INPUT_DISTRICT_FIELD = "detecteddistrict"
INPUT_CITY_FIELD = "detectedcity"
INPUT_DATE_ANNOUNCED_FIELD = "dateannounced"
INPUT_STATUS_CHANGE_DATE_FIELD = "statuschangedate"

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


def load_observations_data():
    raw_data = get_raw_data_dict(raw_data_url)["raw_data"]
    df_list = []
    for region in [(INPUT_DISTRICT_FIELD, DISTRICT), (INPUT_CITY_FIELD, CITY), (INPUT_STATE_FIELD, STATE)]:
        for variable in [CONFIRMED, HOSPITALIZED, RECOVERED, DECEASED, ACTIVE]:
            if variable == ACTIVE:
                df = _get_covid_ts(raw_data, region[0], region[1], HOSPITALIZED)  ## we are adding row for active as hospitalized
            else:
                df = _get_covid_ts(raw_data, region[0], region[1], variable)
            df_list.append(df)
    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True)
    return merged_df


def load_regional_metadata(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)


def get_raw_data_dict(input_url):
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict


def _get_covid_ts(stats, input_region_field, output_region_type_field, variable):
    if variable == CONFIRMED:
        df = pd.DataFrame \
            ([(i[input_region_field].lower().replace(',', ''), i[INPUT_DATE_ANNOUNCED_FIELD]) for i in stats])
    else:
        df = pd.DataFrame(
            [(i[input_region_field].lower().replace(',', ''), i[INPUT_STATUS_CHANGE_DATE_FIELD]) for i in stats if
             i['currentstatus'].lower() == variable])
    df.columns = [REGION_NAME, 'date']
    date_list = pd.date_range(start="2020-01-22", end=datetime.today()).strftime("%d/%m/%Y")
    df_pivot = pd.pivot_table(df, values=REGION_NAME, index=[REGION_NAME], columns=['date']
                              , aggfunc={REGION_NAME: np.count_nonzero}, fill_value=0)
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
    def get_observations_for_region(region_type, region_name):
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

    @staticmethod
    def get_regional_metadata(region_type, region_name, filepath="../data/regional_metadata.json"):
        metadata = load_regional_metadata(filepath)
        for params in metadata["regional_metadata"]:
            if params["region_type"] == region_type and params["region_name"] == region_name:
                return params["metadata"]
