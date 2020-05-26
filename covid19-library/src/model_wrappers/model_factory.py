from entities.model_class import ModelClass
from model_wrappers.base import ModelWrapperBase
from model_wrappers.seir import SEIR
from model_wrappers.seihrd import SEIHRD


class ModelFactory:

    @staticmethod
    def get_model(model_class: ModelClass, model_parameters):
        if model_class.__eq__(ModelClass.SEIR):
            return SEIR(model_parameters)
        elif model_class.__eq__(ModelClass.SEIHRD):
            return SEIHRD(model_parameters)
        else:
            raise Exception("Model Class is not in supported classes {}".format(["SEIR"]))
