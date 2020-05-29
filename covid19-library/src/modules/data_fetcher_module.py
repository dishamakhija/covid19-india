from entities.data_source import DataSource
from data_fetchers.data_fetcher_factory import DataFetcherFactory

class DataFetcherModule(object):

    @staticmethod
    def get_observations_for_region(region_type, region_name, data_source = DataSource.tracker_district_daily, 
                                    smooth = True):
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        return data_fetcher.get_observations_for_region(region_type, region_name)


    @staticmethod
    def get_observations_for_regions(regions, combined_region_name = None, data_source = 'tracker_district_daily',
                                     smooth = True):
        pass
        # df_list = []
        # if combined_region_name is None:
        #     combined_region_name = "_".join(region["region_name"] for region in regions)
        # for region in regions:
        #     df_region = DataFetcherModule.get_observations_for_region(
        #         region["region_type"], region["region_name"], data_source = DataSource.tracker_district_daily, 
        #         smooth = smooth)
        #     df_list.append(df_region)
        # df = pd.concat(df_list)
        # df = df.groupby([OBSERVATION]).sum().reset_index()
        # df.insert(0, column=REGION_NAME, value=combined_region_name)
        # df.insert(1, column=REGION_TYPE, value=region["region_type"])
        # return df

    @staticmethod
    def get_regional_metadata(region_type, region_name, 
    data_source = DataSource.tracker_district_daily, filepath="../data/regional_metadata.json"):
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        return data_fetcher.get_regional_metadata(region_type, region_name, filepath)


if __name__ == "__main__":
    print(DataFetcherModule.get_observations_for_region("district", "pune", "official_data"))
    # print(DataFetcherModule.get_regional_metadata("district","pune","official_data"))
