#!/usr/bin/env python3
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

try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

COVID_S3_BUCKET = "covid-forecast-v1"
COVID_S3_DATA_KEY = "inputs/regional_covid19_observations.csv"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3',region_name='ap-south-1')


#raw input data from cocvid19india.org
raw_data_url = "https://api.covid19india.org/raw_data.json"
INPUT_STATE_FIELD = "detectedstate"
INPUT_DISTRICT_FIELD = "detecteddistrict"
INPUT_CITY_FIELD = "detectedcity"
INPUT_DATE_ANNOUNCED_FIELD = "dateannounced"
INPUT_STATUS_CHANGE_DATE_FIELD = "statuschangedate"

#column headers and static strings used in output CSV
REGION_TYPE = "region_type"
REGION_NAME = "region_name"
STATE = "state"
DISTRICT = "district" 
CITY =  "city"
OBSERVATION =  "observation"
CONFIRMED =  "confirmed"
DECEASED =  "deceased"
RECOVERED =  "recovered"
HOSPITALIZED =  "hospitalized"

def lambda_handler(event, context):
    uuid_for_this_run = str(uuid.uuid4())
    logger.info("Started processing request - ", uuid_for_this_run)
    today = date.today()
    current_date = today.strftime("%Y%m%d")
    logger.info("current_date = " + current_date)
    
    try:
        raw_data = get_raw_data_dict(raw_data_url)["raw_data"]
        df_list = []
        for region in [(INPUT_DISTRICT_FIELD,DISTRICT),(INPUT_CITY_FIELD,CITY),(INPUT_STATE_FIELD,STATE)]:
            for variable in [CONFIRMED,HOSPITALIZED,RECOVERED,DECEASED]:
                df = _get_covid_ts(raw_data,region[0],region[1],variable)
                df_list.append(df)
        merged_df =  pd.concat(df_list)
        #merged_df.to_csv('input_observations.csv', encoding='utf-8')
        
        #TODO -  perform sanity checks 
        #Communicate failure.
        
        csv_buffer = StringIO()
        merged_df.to_csv(csv_buffer)
        s3_client.put_object(Bucket=COVID_S3_BUCKET, Key=COVID_S3_DATA_KEY, Body=csv_buffer.getvalue())
        return _create_output_obj(COVID_S3_BUCKET,COVID_S3_DATA_KEY)
    except Exception as e:
       logger.exception("Fatal error in lambda_handler")

def _create_output_obj(bucket, key):
    return {
        "statusCode": 200,
        "headers": {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Origin": "*",  # Required for CORS support to work
            "Access-Control-Allow-Credentials": True  # Required for cookies, authorization headers with HTTPS
        },
        "body": json.dumps({
                "bucket": bucket,
                "key": key,
                "presignedUrl": s3_client.generate_presigned_url('get_object',
                                                                     Params={
                                                                        'Bucket': bucket,
                                                                        'Key': key
                                                                        })
            })
        }

def get_raw_data_dict(input_url):
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict

def _get_covid_ts(stats,input_region_field,output_region_type_field,variable):
        if variable==CONFIRMED:
            df = pd.DataFrame([(i[input_region_field].lower().replace(',',''),i[INPUT_DATE_ANNOUNCED_FIELD]) for i in stats])
        else:
            df = pd.DataFrame([(i[input_region_field].lower().replace(',',''),i[INPUT_STATUS_CHANGE_DATE_FIELD]) for i in stats if i['currentstatus'].lower()==variable])
        df.columns = [REGION_NAME,'date']
        date_list = pd.date_range(start="2020-01-22",end=datetime.today()).strftime("%d/%m/%Y")
        df_pivot = pd.pivot_table(df,values=REGION_NAME,index=[REGION_NAME],columns=['date'],aggfunc={REGION_NAME: np.count_nonzero},fill_value=0)
        df_pivot_dated = df_pivot.reindex(date_list, axis=1).fillna(0)
        df_final = df_pivot_dated.cumsum(axis=1)
        df_final.insert(0, REGION_TYPE, output_region_type_field)
        df_final.insert(1, OBSERVATION, variable)
        for date_val in date_list:
            datenew = datetime.strptime(date_val, "%d/%m/%Y")
            datenew = datetime.strftime(datenew, "%-m/%-d/%y")
            df_final = df_final.rename(columns = {date_val:datenew})     
        return df_final
        
if __name__ == "__main__":
    lambda_handler(None, None)