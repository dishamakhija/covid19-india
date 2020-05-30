from modules.scenario_forecasting_module import ScenarioForecastingModule

if __name__ == "__main__":
    # ForecastingModule.from_config_file("../config/sample_forecasting_config.json")
    # ModelEvaluator.from_config_file("../config/sample_evaluation_config.json")
    # TrainingModule.from_config_file("../config/sample_training_config.json")
    # ScenarioForecastingModule.from_config_file("config/sample_npi_scenario_forecasting_config.json")
    ScenarioForecastingModule.from_config_file("../config/sample_param_override_scenario_forecasting_config.json")
