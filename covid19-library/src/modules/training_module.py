from src.model_wrappers.model_factory import ModelFactory


class TrainingModule(object):

    def __init__(self, model_class, model_parameters):
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def train(self, train_start_date, train_end_date,
              region_name, search_space, search_parameters, loss_function, output_path):
        if self._model.is_black_box():

