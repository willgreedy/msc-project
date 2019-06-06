import json


class ParameterConfig:
    def __init__(self, config_filename):
        with open(config_filename, 'r') as f:
            self.parameters = json.load(f)

    def get_parameters(self):
        return self.parameters
