import numpy as np


def create_transfer_function(config):
    type = config['type']
    if type == 'soft-recify':
        gamma = config['gamma'] if 'gamma' in config else 1
        beta = config['beta'] if 'beta' in config else 1
        theta = config['theta'] if 'theta' in config else 0

        return lambda u: gamma * np.log(1 + np.exp(beta * (u - theta)))