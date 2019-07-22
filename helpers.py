
from abc import ABC, abstractmethod

import numpy as np
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
import pickle
import os
import pathlib
import shutil


def create_transfer_function(config, with_gradient=False):
    type = config['type']
    if type == 'soft-rectify':
        gamma = config['gamma'] if 'gamma' in config else 1.0
        beta = config['beta'] if 'beta' in config else 1.0
        theta = config['theta'] if 'theta' in config else 0.0

        def transfer_fun(u):
            inner_vals = beta * (u.copy() - theta)
            result = np.zeros(u.shape)
            result[inner_vals < 500.0] = gamma * np.log(1.0 + np.exp(inner_vals[inner_vals < 500.0]))
            result[inner_vals >= 500.0] = gamma * inner_vals[inner_vals > 500.0]
            return result

        if with_gradient:
            def gradient_fun(u):
                inner_vals = beta * (u.copy() - theta)
                grads = np.zeros(u.shape)
                grads[inner_vals < -500.0] = 0.0
                grads[inner_vals >= -500.0] = gamma * beta / (1.0 + np.exp(-inner_vals[inner_vals >= -500.0]))
                return grads

    elif type == 'logistic':
        def transfer_fun(u):
            result = 1.0 / (1.0 + np.exp(-u))
            return result
        if with_gradient:
            def gradient_fun(u):
                res = transfer_fun(u)
                return res * (1 - res)
    else:
        raise Exception("Invalid transfer function: {}".format(type))

    if with_gradient:
        return transfer_fun, gradient_fun
    else:
        return transfer_fun


def remove_directory(location):
    location = pathlib.Path(location)
    shutil.rmtree(location, ignore_errors=True)


def visualise_mnist(image):
    plt.figure()
    plt.imshow(image.reshape(28,28), cmap='Greys')


def visualise_transfer_function(transfer_function, gradient_function=None):
    xs = np.arange(-1000.0, 1000.0, 0.01)
    ys = transfer_function(xs)
    plt.figure()
    plt.plot(xs, ys)
    if gradient_function is not None:
        ys2 = gradient_function(xs)
        plt.plot(xs, ys2)
    plt.title("Transfer function")
    plt.show()


def show_plots():
    plt.show()


def get_target_network_forward_weights_list(target_network_weights_path):
    target_network_forward_weights_list = []
    i = 1
    while True:
        target_network_weight_path = "{}/layer{}_weights.npy".format(target_network_weights_path, i)
        if os.path.exists(target_network_weight_path):
            target_network_forward_weights_list += [np.load(target_network_weight_path).copy()]
            print("Target Network layer {} weights: {}".format(i, target_network_forward_weights_list[-1]))
        else:
            break
        i += 1
    return target_network_forward_weights_list


def compute_non_linear_transform(input_sequence, transfer_function, feedforward_weights_list=list()):
    curr_values = input_sequence
    for feedforward_weights in feedforward_weights_list[:-1]:
        curr_values = np.matmul(curr_values, feedforward_weights)
        print("Linear: {}+-{}, Max={}".format(np.mean(curr_values), np.std(curr_values), np.max(curr_values)))
        curr_values = transfer_function(curr_values)
        print("Transfer: {}+-{}, %Large={}, Max={}".format(np.mean(curr_values), np.std(curr_values), (curr_values > 0.5).sum() / len(curr_values.flatten()), np.max(curr_values)))

    curr_values = np.matmul(curr_values, feedforward_weights_list[-1])
    print("Linear: {}+-{}, Max={}".format(np.mean(curr_values), np.std(curr_values), np.max(curr_values)))
    return curr_values


def load_model(location):
    pkl_file = open(location, 'rb')
    return pickle.load(pkl_file)


def save_model(save_location, name, model):
    output = open(save_location + '/' + name + '.pkl', 'wb')
    pickle.dump(model, output)


def read_monitoring_values_config_file(config_file):
    return [line.rstrip('\n') for line in open(config_file)]


class Initialiser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, shape):
        pass


class UniformInitialiser(Initialiser):
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self, shape):
        res = np.random.uniform(self.lower_bound, self.upper_bound, shape)
        #print("Result: {}".format(res))
        return res


class ConstantInitialiser(Initialiser):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def sample(self, shape):
        return np.ones(shape) * self.value
