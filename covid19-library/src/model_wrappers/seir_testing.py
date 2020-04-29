import pandas as pd
from model_wrappers.base import ModelWrapperBase


class SEIR_Testing(ModelWrapperBase):

    def __init__(self):
        pass

    def supported_forecast_variables(self):
        pass

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, 
                      forecast_start_date: str, forecast_end_date: str, **kwargs):
        pass

    def fit(self):
        pass

    def is_black_box(self):
        return True


"""
Lines following this will be removed upon completion of this model class.
This is just for easy debugging.
"""
if __name__ == '__main__':
    import pdb
    from modules.data_fetcher_module import DataFetcherModule

    region_type = 'district'
    region_name = 'bengaluru'
    run_day = '3/22/20'
    forecast_start_date = '4/1/20'
    forecast_end_date = '4/20/20'

    region_observations = DataFetcherModule.get_observations_for_region(region_type, region_name)
    regional_metadata_path = '/home/users/namrata/projects/covid19-india/covid19-library/data/regional_metadata.json'
    region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, filepath=regional_metadata_path)

    model = SEIR_Testing()
    predictions_df = model.predict(region_metadata, region_observations, run_day, forecast_start_date, forecast_end_date)

    pdb.set_trace()

    print('abc')