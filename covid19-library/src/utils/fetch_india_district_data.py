import requests
import json
import pandas as pd
import datetime
import numpy as np

covid_india_url = "https://api.covid19india.org/raw_data.json"
start_date = "2020-01-22"


def _request(method, path, data={}, parameters={}, headers={}):
    response = requests.request(method, path, params=parameters, json=data, headers=headers)
    response.raise_for_status()
    return response.json()


def convert_to_jhu_format(records_df, REGION='Province/State'):
    date_list = pd.date_range(start=start_date, end=datetime.datetime.today()).strftime("%d/%m/%Y")
    district_df_pivot = pd.pivot_table(records_df, values=REGION, index=[REGION], columns=['date'],
                                       aggfunc={REGION: np.count_nonzero}, fill_value=0)
    district_df_pivot_dated = district_df_pivot.reindex(date_list, axis=1).fillna(0)
    district_df_final = district_df_pivot_dated.cumsum(axis=1)
    district_df_final.insert(0, "Country/Region", "India")
    district_df_final.insert(1, "Lat", 20)
    district_df_final.insert(2, "Long", 70)
    formatted_date = [datetime.datetime.strptime(date, "%d/%m/%Y").strftime("%-m/%-d/%y") for date in date_list]
    district_df_final = district_df_final.rename(columns=dict(zip(date_list, formatted_date)))
    district_df_final.reset_index(inplace=True)
    return district_df_final


# data_type can be confirmed, Hospitalized, Recovered, Deceased
def get_india_district_data_from_url(region_name, data_type="confirmed"):
    records = _request("GET", covid_india_url)['raw_data']
    records_df = pd.DataFrame.from_records(records)
    if data_type == "confirmed":
        records_df = records_df[["detecteddistrict", "dateannounced"]]
    elif data_type in ["Hospitalized", "Recovered", "Deceased"]:
        records_df = records_df[records_df.currentstatus == data_type]
        records_df = records_df[["detecteddistrict", "statuschangedate"]]
    else:
        raise BaseException("data type is not supported")
    REGION = 'Province/State'
    records_df.columns = [REGION, 'date']
    nhu_df = convert_to_jhu_format(records_df, REGION)
    return nhu_df[nhu_df[REGION] == region_name].iloc[0]


# def get_india_district_data_all(region_name):


def get_population_value(region_name, filepath="../data/population_data.csv"):
    df = pd.read_csv(filepath)
    return df[df.region_name == region_name]["population"].values[0]


if __name__ == "__main__":
    confirmed_data = get_india_district_data_from_url("Bengaluru", "confirmed")
    recovered_data = get_india_district_data_from_url("Bengaluru", "Recovered")
    print(confirmed_data)
    print(recovered_data)
