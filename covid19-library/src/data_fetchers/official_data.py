from entities.data_source import DataSource
from data_fetchers.data_fetcher_base import DataFetcherBase

import copy
import numpy as np
import pandas as pd

from pathlib import Path

from pyathena import connect
from pyathena.pandas_cursor import PandasCursor

def create_connection(pyathena_rc_path=None):
    """Creates SQL Server connection using AWS Athena credentials
    Keyword Arguments:
        pyathena_rc_path {str} -- [Path to the PyAthena RC file with the AWS Athena variables] (default: {None})
    Returns:
        [cursor] -- [Connection Cursor]
    """
    if pyathena_rc_path == None:
        pyathena_rc_path = Path(__file__).parent / "../../../../pyathena/pyathena.rc"
    SCHEMA_NAME = 'wiai-covid-data'

    # Open Pyathena RC file and get list of all connection variables in a processable format
    with open(pyathena_rc_path) as f:
        lines = f.readlines()

    lines = [x.strip() for x in lines]
    lines = [x.split('export ')[1] for x in lines]
    lines = [line.replace('=', '="') + '"' if '="' not in line else line for line in lines]
    variables = [line.split('=') for line in lines]

    # Create variables using the processed variable names from the RC file
    AWS_CREDS = {}
    for key, var in variables:
        exec("{} = {}".format(key, var), AWS_CREDS)

    # Create connection
    cursor = connect(aws_access_key_id=AWS_CREDS['AWS_ACCESS_KEY_ID'],
                     aws_secret_access_key=AWS_CREDS['AWS_SECRET_ACCESS_KEY'],
                     s3_staging_dir=AWS_CREDS['AWS_ATHENA_S3_STAGING_DIR'],
                     region_name=AWS_CREDS['AWS_DEFAULT_REGION'],
                     work_group=AWS_CREDS['AWS_ATHENA_WORK_GROUP'],
                     schema_name=SCHEMA_NAME).cursor(PandasCursor)
    return cursor

def get_athena_dataframes(pyathena_rc_path=None):
    """Creates connection to Athena database and returns all the tables there as a dict of Pandas dataframes
    Keyword Arguments:
        pyathena_rc_path {str} -- Path to the PyAthena RC file with the AWS Athena variables 
        (default: {None})
    Returns:
        dict -- dict where key is str and value is pd.DataFrame
        The dataframes : 
        covid_case_summary
        demographics_details
        healthcare_capacity
        testing_summary
    """
    if pyathena_rc_path == None:
        pyathena_rc_path = Path(__file__).parent / "../../../../pyathena/pyathena.rc"

    # Create connection
    cursor = create_connection(pyathena_rc_path)

    # Run SQL SELECT queries to get all the tables in the database as pandas dataframes
    dataframes = {}
    tables_list = cursor.execute('Show tables').as_pandas().to_numpy().reshape(-1, )
    for table in tables_list:
        dataframes[table] = cursor.execute(
            'SELECT * FROM {}'.format(table)).as_pandas()
    
    return dataframes

def get_data_from_db(district):
    dataframes = get_athena_dataframes()
    df_result = copy.copy(dataframes['covid_case_summary'])
    df_result = df_result[df_result['district'] == district.lower()]
    df_result['date'] = pd.to_datetime(df_result['date']).apply(lambda x: x.strftime("%-m/%-d/%y"))

    del df_result['state']
    del df_result['district']
    del df_result['ward_name']
    del df_result['ward_no']
    del df_result['mild']
    del df_result['moderate']
    del df_result['severe']
    del df_result['critical']
    del df_result['partition_0']

    df_result.columns = [x if x != 'active' else 'hospitalized' for x in df_result.columns]
    df_result.columns = [x if x != 'total' else 'confirmed' for x in df_result.columns]
    df_result = df_result.fillna(0)

    df_result = df_result.rename(columns={'date':'index'})
    df_result = df_result.set_index('index').transpose().reset_index().rename(columns={'index':"observation"})
    df_result.insert(0, column = "region_name", value = district.lower().replace(',', ''))
    df_result.insert(1, column = "region_type", value = "district")

    return df_result


class OfficialData(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name):
        if region_type != 'district':
            raise NotImplementedError
        return get_data_from_db(region_name)