import numpy as np
from abc import ABC
import matplotlib.pyplot as plt
import pickle


def create_transfer_function(config):
    type = config['type']
    if type == 'soft-rectify':
        gamma = config['gamma'] if 'gamma' in config else 1.0
        beta = config['beta'] if 'beta' in config else 1.0
        theta = config['theta'] if 'theta' in config else 0.0

        def transfer_fun(u):
            result = beta * (u - theta)
            result[result < 500.0] = gamma * np.log(1.0 + np.exp(result[result<500.0]))
            return result

        return transfer_fun
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
    plt.xlim(iter_numbers[0], iter_numbers[-1])
    plt.title(monitor.get_var_name())


def visualise_MNIST(image):
    plt.figure()
    plt.imshow(image.reshape(28,28), cmap='Greys')


def visualise_transfer_function(transfer_function):
    xs = np.arange(-100.0, 100.0, 0.01)
    ys = transfer_function(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.title("Transfer function")


def show_plots():
    plt.show()


def compute_non_linear_transform(input_sequence, transfer_function, feedforward_weights_list=list()):
    curr_values = input_sequence
    for feedforward_weights in feedforward_weights_list:
        curr_values = transfer_function(curr_values)
        print("Transfer: {}".format(curr_values))
        curr_values = np.matmul(curr_values, feedforward_weights)
        print("Linear: {}".format(curr_values))

    #curr_values = np.matmul(curr_values, feedforward_weights_list[-1])
    #print("Linear: {}".format(curr_values))
    return curr_values


def load_model(name):
    pkl_file = open('saved_models/' + name + '.pkl', 'rb')
    return pickle.load(pkl_file)


def save_model(name, model):
    output = open('saved_models/' + name + '.pkl', 'wb')
    pickle.dump(model, output)


class Initialiser(ABC):
    def __init__(self):
        pass


class UniformInitialiser(Initialiser):
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self, shape):
        res = np.random.uniform(self.lower_bound, self.upper_bound, shape)
        print("Result: {}".format(res))
        return res


class ConstantInitialiser(Initialiser):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def sample(self, shape):
        return np.ones(shape) * self.value