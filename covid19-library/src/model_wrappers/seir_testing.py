from model_wrappers.base import ModelWrapperBase


class SEIR_Testing(ModelWrapperBase):

	def __init__(self):
		pass

    def supported_forecast_variables(self):
        pass

    def predict(self, confirmed_data: pd.Series, recovered_data: pd.Series, run_day, start_date, end_date, **kwargs):
        pass

    def fit(self):
        pass

    def is_black_box(self):
        return True