from src.entities.model_class import ModelClass
from src.model_wrappers.base import ModelWrapperBase
from src.model_wrappers.seir import SEIR


class ModelFactory:

    @staticmethod
    def get_model(model_class: ModelClass, model_parameters):
        if model_class.__eq__(ModelClass.SEIR):
            return SEIR(model_parameters)
        else:
            raise Exception("Model Class is not in supported classes {}".format(["SEIR"]))
