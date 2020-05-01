#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import boto3
import shutil
import uuid
import datetime
import json
import io
from io import StringIO
import pandas as pd
import requests
from datetime import date

from modules.forecasting_module import ForecastingModule

COVID_S3_BUCKET = "covid-forecast-v1"
REGIONAL_MODEL_PARAMS_KEY = "inputs/regional_model_params.json"
REGIONAL_METADATA_KEY = "inputs/regional_metadata.json"
REGIONAL_COVID19_OBSERVATIONS_KEY = "inputs/regional_covid19_observations.csv"
OUTPUT_CSV_KEY = "covid_forecast_output_v1.csv"
OUTPUT_JSON_KEY = "covid_forecast_output_v1.json"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3', region_name='ap-south-1')

output = {
    "statusCode": 200,
    "headers": {
        'Content-Type': 'application/json',
        "Access-Control-Allow-Origin": "*",  # Required for CORS support to work
        "Access-Control-Allow-Credentials": True  # Required for cookies, authorization headers with HTTPS
    },
    "body": json.dumps({
        "status": "SUCCESS"
    }),
}

testInput = {
    "body": json.dumps({
        "regions": [
            {
                "region_name": "mumbai",
                "region_type": "district",
                "model_name": "SEIR",
            },
            {
                "region_name": "pune",
                "region_type": "district",
                # TODO: composite key to include
                "model_name": "SEIR",
            },
            {
                "region_name": "bengaluru",
                "region_type": "district",
                "model_name": "SEIR",
            },
            # {
            #        "region_name":"bengaluru rural",
            #        "region_type":"district",
            #        "model_name":"SEIR",
            # }
            {
                "region_name": "west bengal",
                "region_type": "state",
                "model_name": "SEIR",
            },
            {
                "region_name": "delhi",
                "region_type": "state",
                "model_name": "SEIR",
            },
            {
                "region_name": "maharashtra",
                "region_type": "state",
                "model_name": "SEIR",
            },
            {
                "region_name": "karnataka",
                "region_type": "state",
                "model_name": "SEIR",
            }

        ]
    }),
}


def lambda_handler(event, context):
    logger.info("Started processing request ")
    uuid_for_this_run = str(uuid.uuid4())
    logger.info("uuid_for_this_run: " + uuid_for_this_run)
    try:
        metadata = _load_regional_metadata_from_S3(COVID_S3_BUCKET, REGIONAL_METADATA_KEY)
        model_params = _load_regional_model_params_from_s3(COVID_S3_BUCKET, REGIONAL_MODEL_PARAMS_KEY)
        observations = _load_regional_observations_from_s3(COVID_S3_BUCKET, REGIONAL_COVID19_OBSERVATIONS_KEY)
        input_obj = json.loads(event["body"])
        logger.info("Received input " + str(input_obj))

        df_list = []
        regions = input_obj["regions"]
        for regionParams in regions:
            region_name = regionParams["region_name"]
            region_type = regionParams["region_type"]
            model_name = regionParams["model_name"]
            df = _predict_for_region(region_name, region_type, model_name, model_params, metadata, observations)
            df_list.append(df)

        output_df = pd.concat(df_list)

        csv_buffer = StringIO()
        output_df.to_csv(csv_buffer)
        s3_client.put_object(Bucket=COVID_S3_BUCKET, Key=OUTPUT_CSV_KEY, Body=csv_buffer.getvalue())

        json_data = output_df.to_json(orient='records')
        s3_client.put_object(Bucket=COVID_S3_BUCKET, Key=OUTPUT_JSON_KEY,
                             Body=(bytes(json.dumps(json_data).encode('UTF-8'))))

        return output
    except Exception as e:
        logger.exception("Something failed {}", e)
        raise e


def _predict_for_region(region_name, region_type, model_name, model_params, metadata, observations):
    print("prediction - ", region_name, region_type, model_name)

    # fetch model params
    model_params = _get_model_params_for_region(region_type, region_name, model_name, model_params)

    # fetch region metadata
    region_metadata = _get_metadata_for_region(region_type, region_name, metadata)

    # fetch filtered timeseries
    region_observations = _get_observations_for_region(region_type, region_name, observations)
    # todo - print some debug metadata on generated outputs
    forecasting_module = ForecastingModule(model_name, model_params)
    current_date = date.today()
    forecast_start_date = current_date.strftime("%-m/%-d/%y")
    forecast_end_date = (current_date + datetime.timedelta(days=30)).strftime("%-m/%-d/%y")
    run_day = (current_date - datetime.timedelta(days=1)).strftime(
        "%-m/%-d/%y")  ##initializing run day as current date -1
    output_df = forecasting_module.predict(region_type, region_name, region_metadata, region_observations, run_day,
                                           forecast_start_date,
                                           forecast_end_date)
    return output_df


# def _invoke_seir(region_name,region_type,model_params,metadata,observations):
#     print("seir - ",region_name,region_type)
#
#     model = SEIRSModel(
#         initN=metadata["population"], #
#         beta=model_params["beta"],
#         sigma=model_params["sigma"],
#         gamma=model_params["gamma"],
#         initE=model_params["init_E"],
#         )
#     model.run(T=300)
#     print(model.numI)
#     print(model.tseries)
#
#     #dummy df
#     column_names = ["region_type", "region_name", "variable", "date1", "date2"]
#     return pd.DataFrame(columns = column_names)

def _invoke_unknown(region_name, region_type, model_params, metadata, observations):
    raise Exception("Unknown model for {} {}".format(region_name, region_type))


def _get_model_params_for_region(region_type, region_name, model_name, model_params):
    for params in model_params["regional_model_params"]:
        if params["region_type"] == region_type and params["region_name"] == region_name and params[
            "model_name"] == model_name:
            return params["model_params"]


def _get_metadata_for_region(region_type, region_name, metadata):
    for params in metadata["regional_metadata"]:
        if params["region_type"] == region_type and params["region_name"] == region_name:
            return params["metadata"]


def _get_observations_for_region(region_type, region_name, observations_df):
    temp_df = observations_df[observations_df["region_name"] == region_name]
    return temp_df[temp_df["region_type"] == region_type]


def _load_regional_metadata_from_S3(bucket, key):
    return _read_json_from_s3(bucket, key)


def _load_regional_model_params_from_s3(bucket, key):
    return _read_json_from_s3(bucket, key)


def _read_json_from_s3(bucket, key):
    s3_obj = s3_client.get_object(Bucket=bucket, Key=key)
    s3_data = s3_obj['Body'].read().decode('utf-8')
    return json.loads(s3_data)


def _load_regional_observations_from_s3(bucket, key):
    return _read_csv_from_s3(bucket, key)


def _read_csv_from_s3(bucket, key):
    csv_obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))


if __name__ == "__main__":
    lambda_handler(testInput, None)
