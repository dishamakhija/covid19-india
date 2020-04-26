from functools import partial
import json
from typing import List

from hyperopt import hp

from src.configs.base_config import TrainingModuleConfig
from src.entities.loss_function import LossFunction
from src.model_wrappers.model_factory import ModelFactory
from src.modules.model_evaluator import ModelEvaluator
from src.utils.config_util import read_config_file
from src.utils.fetch_india_district_data import get_india_district_data_from_url, get_population_value
from src.utils.hyperparam_util import hyperparam_tuning


class TrainingModule(object):

    def __init__(self, model_class, model_parameters):
        self._model = ModelFactory.get_model(model_class, model_parameters)
        self._model_class = model_class
        self._model_parameters = model_parameters

    def train(self, train_start_date, train_end_date,
              region_name, search_space, search_parameters, train_loss_function):
        confirmed_data = get_india_district_data_from_url(region_name, "confirmed")
        recovered_data = get_india_district_data_from_url(region_name, "Recovered")
        population = get_population_value(region_name)
        confirmed_data["population"] = population
        result = {}
        if self._model.is_black_box():
            objective = partial(self.optimize, confirmed_data=confirmed_data, recovered_data=recovered_data,
                                train_start_date=train_start_date,
                                train_end_date=train_end_date, loss_function=train_loss_function)
            for k, v in search_space.items():
                search_space[k] = hp.uniform(k, v[0], v[1])
            result = hyperparam_tuning(objective, search_space,
                                       search_parameters.get("max_evals", 100))
        return result

    def optimize(self, search_space, confirmed_data, recovered_data, train_start_date, train_end_date,
                 loss_function):
        predict_df = self._model.predict(confirmed_data, recovered_data, train_start_date, train_start_date,
                                         train_end_date,
                                         search_space)
        metrics_result = ModelEvaluator.evaluate_for_forecast(predict_df, [loss_function])
        return metrics_result[0]["value"]

    @staticmethod
    def get_train_loss_function(loss_functions: List[LossFunction]):
        result = list(filter(lambda x: x.optimize == True, loss_functions))
        if len(result) == 0:
            raise Exception("No loss function declared with optimize as true ")
        return result[0]  # if multiple functions are declared as optimize true, taking first one

    @staticmethod
    def from_config(config: TrainingModuleConfig):
        training_module = TrainingModule(config.model_class, config.model_parameters)
        train_loss_func = TrainingModule.get_train_loss_function(config.loss_functions)
        results = {}
        training_results = training_module.train(config.train_start_date, config.train_end_date, config.region_name,
                                                 config.search_space,
                                                 config.search_parameters, train_loss_func)
        results.update(training_results)
        config.model_parameters.update(
            training_results["best_params"])  # updating model parameters with best params found above
        model_evaluator = ModelEvaluator(config.model_class, config.model_parameters)
        results["train_metric_results"] = model_evaluator.evaluate(config.region_name, config.train_start_date,
                                                                   config.train_start_date, config.train_end_date,
                                                                   config.loss_functions)
        if config.test_start_date is not None:
            results["test_metric_results"] = model_evaluator.evaluate(config.region_name, config.train_start_date,
                                                                      config.test_start_date, config.test_end_date,
                                                                      config.loss_functions)
        if config.output_filepath is not None:
            with open(config.output_filepath, 'w') as outfile:
                json.dump(results, outfile)
        return results

    @staticmethod
    def from_config_file(config_file_path: str):
        config = read_config_file(config_file_path)
        training_module_config = TrainingModuleConfig.parse_obj(config)
        return TrainingModule.from_config(training_module_config)
