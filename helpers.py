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


def create_diff_plot(monitor1, monitor2):
    iter_numbers, values1 = monitor1.get_values()
    _, values2 = monitor2.get_values()
    print("Creating diff plot with {} values.".format(len(values1)))
    plt.figure()
    plt.plot(iter_numbers, np.array(values1) - np.array(values2))
    plt.title(monitor1.get_var_name() + "-" + monitor2.get_var_name())


def create_plot(monitor):
    iter_numbers, values = monitor.get_values()
    print("Creating plot with {} values.".format(len(values)))
    plt.figure()
    plt.plot(iter_numbers, values)
    plt.title(monitor.get_var_name())


def visualise_MNIST(image):
    plt.figure()
    plt.imshow(image.reshape(28,28), cmap='Greys')


def visualise_transfer_function(transfer_function):
    xs = np.arange(-2.0, 2.0, 0.01)
    ys = transfer_function(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.title("Transfer function")

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

