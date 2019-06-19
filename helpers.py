import numpy as np
from abc import ABC
import matplotlib.pyplot as plt


def create_transfer_function(config):
    type = config['type']
    if type == 'soft-rectify':
        gamma = config['gamma'] if 'gamma' in config else 1
        beta = config['beta'] if 'beta' in config else 1
        theta = config['theta'] if 'theta' in config else 0

        return lambda u: gamma * np.log(1 + np.exp(beta * (u - theta)))
    else:
        raise Exception("Invalid transfer function: {}".format(type))


def create_plot(monitor):
    iter_numbers, values = monitor.get_values()
    plt.figure()
    plt.plot(iter_numbers, values)
    plt.title(monitor.get_var_name())


def create_MNIST_visualisation(image):
    plt.figure()
    plt.imshow(image.reshape(28,28), cmap='Greys')

def show_plots():
    plt.show()


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

