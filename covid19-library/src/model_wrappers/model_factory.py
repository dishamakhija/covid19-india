from entities.model_class import ModelClass
from model_wrappers.base import ModelWrapperBase
from model_wrappers.intervention_enabled_seihrd import InterventionEnabledSEIHRD
from model_wrappers.intervention_enabled_seir import InterventionEnabledSEIR
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

    @staticmethod
    def get_intervention_enabled_model(model_class: ModelClass, model_parameters):
        if model_class.__eq__(ModelClass.SEIR):
            return InterventionEnabledSEIR(model_parameters)
        if model_class.__eq__(ModelClass.SEIHRD):
            return InterventionEnabledSEIHRD(model_parameters)
        else:
            raise Exception("Model Class is not in supported classes {}".format(["SEIR"]))
