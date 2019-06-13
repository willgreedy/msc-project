import numpy as np
from abc import ABC


def create_transfer_function(config):
    type = config['type']
    if type == 'soft-rectify':
        gamma = config['gamma'] if 'gamma' in config else 1
        beta = config['beta'] if 'beta' in config else 1
        theta = config['theta'] if 'theta' in config else 0

        return lambda u: gamma * np.log(1 + np.exp(beta * (u - theta)))
    else:
        raise Exception("Invalid transfer function: {}".format(type))


class Initialiser(ABC):
    def __init__(self):
        pass


class UniformInitialiser(Initialiser):
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self, shape):
        return np.random.uniform(self.lower_bound, self.upper_bound, shape)
