import json


class ParameterConfig:
    def __init__(self, config_filename='./parameter_configs/default.json'):
        with open(config_filename, 'r') as f:
            self.parameters = json.load(f)

    def get_parameters(self):
        return self.parameters

    def __getitem__(self, item):
        return self.parameters.__getitem__(item)
