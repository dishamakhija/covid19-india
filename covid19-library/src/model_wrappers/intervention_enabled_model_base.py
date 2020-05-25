from model_wrappers.base import ModelWrapperBase


class InterventionEnabledModelBase(ModelWrapperBase):

    @property
    @abc.abstractmethod
    def supported_parameters_override(self):
        """
        @return: list of parameters which can be overridden for interventions
        @rtype: list of strings
        """
        pass

    @property
    @abc.abstractmethod
    def get_scale_factor_map(self):
        """
        @return: map of scale factor
        @rtype: dict of <string, float>
        """
        pass
