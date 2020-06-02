from collections import defaultdict

import pandas as pd

from entities.intervention_variable import InterventionVariable, InputType
from model_wrappers.base import ModelWrapperBase
import abc


class InterventionEnabledModelBase(ModelWrapperBase):

    def __init__(self, model_parameters):
        self.model_parameters = model_parameters

    @abc.abstractmethod
    def get_supported_parameters_override(self):
        """
        :return: list of parameters which can be overridden for interventions
        :rtype: list of strings
        """
        pass

    @abc.abstractmethod
    def get_base_model(self):
        """
        :return: return the object base model for this intervention enabled model
        :rtype: ModelWrapperBase
        """

    @property
    def get_supported_intervention_variables(self):
        """
        :return: list of intervention variables which are supported by this model.
        By default, it returns all intervention variables
        :rtype: list of strings
        """
        return [variable for variable in InterventionVariable]

    @abc.abstractmethod
    def get_scale_factor_map(self):
        """ intervention effect parameters
        :return: map of scale factor
        :rtype: dict<string, float>
        """
        pass

    @abc.abstractmethod
    def get_change_factor_map(self):
        """ intervention-model parameter change expressions
        :return: returns map of parameters, and
        lambda for calculation
        :rtype: dict<string, lambda>
        """
        pass

    def compute_change_factors(self, intervention_map: dict):
        """
        :param intervention_map: dict of intervention variable and value
        :return: returns the model parameter vs factor map after applying interventions
        :rtype: dict<string, float>
        """
        self.validate_intervention_map(intervention_map)
        param_change_factor_map = {}
        for intervention in intervention_map:
            level = intervention_map[intervention]
            for (param, expr) in self.get_change_factor_map[intervention]:
                param_change_factor_map[param] = param_change_factor_map.get(param, 1) * expr(level, None)
        return param_change_factor_map

    def validate_intervention_map(self, intervention_map: dict):
        """
        :param intervention_map: dict of intervention variable and value
        :return: bool, true if all variables are supported by the model
        """
        variables = set(intervention_map.keys())
        not_supported_variables = variables - set(self.get_supported_intervention_variables)
        if len(not_supported_variables) == 0:
            return True
        raise Exception("Intervention variables not supported: %s".format(str(not_supported_variables)))

    def validate_params_for_override(self, params_map):
        """

        :param params_map: value of overridden param and its value
        :return: bool, true if the overridden params are supported by the model
        """
        params = set(params_map.keys())
        not_supported_params = params - set(self.get_supported_parameters_override)
        if len(not_supported_params) == 0:
            return True
        raise Exception("Param not supported for override: %s".format(str(not_supported_params)))

    def apply_nhi_interventions(self, intervention_map):
        """ updates model parameters after applying NHI interventions
        :param intervention_map: dict of intervention variable and value
        :return: None
        """
        self.validate_intervention_map(intervention_map)
        param_change_factor_map = self.compute_change_factors(intervention_map)
        for key in param_change_factor_map.keys():
            self.model_parameters[key] *= param_change_factor_map.get(key, 1)

    def apply_params_override(self, params_map):
        """ updates model parameters after applying overridden params
        :param params_map: dict of parameters and their overridden values
        :return: None
        """
        self.validate_params_for_override(params_map)
        self.model_parameters.update(params_map)

    @property
    def supported_forecast_variables(self):
        return self.get_base_model().supported_forecast_variables

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        return self.get_base_model().predict(region_metadata, region_observations, run_day, start_date, end_date,
                                             **kwargs)

    def fit(self):
        raise BaseException("Not supported for intervention enabled models")

    def is_black_box(self):
        return self.get_base_model().is_black_box()

    def predict_for_scenario(self, input_type: InputType, interventions_map: dict, region_metadata: dict,
                             region_observations: pd.DataFrame, run_day: str, start_date: str,
                             end_date: str):
        if input_type.__eq__(InputType.param_override):
            self.apply_params_override(interventions_map)
        elif input_type.__eq__(InputType.npi_list):
            self.apply_nhi_interventions(interventions_map)
        else:
            raise Exception("InputType is not implemented")
        return self.predict(region_metadata, region_observations, run_day, start_date, end_date)
