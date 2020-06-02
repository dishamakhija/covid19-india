import pandas as pd

from model_wrappers.intervention_enabled_model_base import InterventionEnabledModelBase
from model_wrappers.seihrd import SEIHRD


class InterventionEnabledSEIHRD(InterventionEnabledModelBase):

    def __init__(self, model_parameters: dict):
        super().__init__(model_parameters)

    @property
    def get_supported_parameters_override(self):
        params = ["r0", "incubation_period", "infectious_period"]
        return params

    def get_base_model(self):
        return SEIHRD(self.model_parameters)

    @property
    def get_change_factor_map(self):
        change_factor_map = {
            'testing_rate': [('incubation_period',
                              lambda x, y: 1 - x * self.get_scale_factor_map['TESTING_TINC_DROP']),
                             ('infectious_period',
                              lambda x, y: 1 - x * self.get_scale_factor_map['TESTING_TINF_DROP'])],
            'containment_fraction': [('infectious_period',
                                      lambda x, y: 1 - x * self.get_scale_factor_map['CONTAINMENT_TINF_DROP']),
                                     ('r0',
                                      lambda x, y: 1 - x * self.get_scale_factor_map['CONTAINMENT_R0_DROP'])],
            'mask_compliance': [('r0',
                                 lambda x, y: 1 - x * self.get_scale_factor_map['MASK_R0_DROP'])]
        }
        return change_factor_map

    @property
    def get_scale_factor_map(self):
        return {
            'MASK_R0_DROP': 0.1,
            'CONTAINMENT_R0_DROP': 0.2,
            'TESTING_TINC_DROP': 0.3,
            'TESTING_TINF_DROP': 0.4,
            'CONTAINMENT_TINF_DROP': 0.5
        }
