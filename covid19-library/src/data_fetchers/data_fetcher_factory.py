from entities.data_source import DataSource
from data_fetchers.data_fetcher_base import DataFetcherBase
from data_fetchers.tracker_raw import TrackerRaw
from data_fetchers.tracker_district_daily import TrackerDistrictDaily
from data_fetchers.official_data import OfficialData

class DataFetcherFactory:

    @staticmethod
    def get_data_fetcher(data_source: DataSource):
        if data_source.__eq__(DataSource.tracker_raw_data):
            return TrackerRaw()
        elif data_source.__eq__(DataSource.tracker_district_daily):
            return TrackerDistrictDaily()
        elif data_source.__eq__(DataSource.official_data):
            return OfficialData()
        else:
            raise Exception("Data source is not in supported sources.")