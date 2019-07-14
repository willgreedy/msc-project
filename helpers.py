import numpy as np
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pickle
import pathlib
import shutil
import datetime

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

    elif type == 'logistic':
        def transfer_fun(u):
            result = 1.0 / (1.0 + np.exp(-u))
            return result
    else:
        raise Exception("Invalid transfer function: {}".format(type))
    return transfer_fun


def create_diff_plot(monitor1, monitor2):
    iter_numbers, values1 = monitor1.get_values()
    _, values2 = monitor2.get_values()
    print("Creating diff plot with {} values.".format(len(values1)))
    plt.figure()
    plt.plot(iter_numbers, np.array(values1) - np.array(values2))
    plt.title(monitor1.get_var_name() + "-" + monitor2.get_var_name())

def create_plot(monitor, save_location=None, close_plot=True):
    iter_numbers, values = monitor.get_values()
    print("Creating plot with {} values.".format(len(values)))
    fig, ax = plt.subplots()
    ax.plot(iter_numbers, values)
    ax.set_xlim(iter_numbers[0], iter_numbers[-1])
    y_range = monitor.get_plot_range()
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    ax.set_title(monitor.get_var_name())
    if save_location is not None:
        time = datetime.datetime.now().strftime("%I-%M%p %B%d")
        filename = '/{}'.format(monitor.get_name())

        pathlib.Path(save_location).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_location + filename + ".pdf", bbox_inches='tight')

        plot_objects_location = save_location + '/plot_objects/'
        pathlib.Path(plot_objects_location).mkdir(parents=True, exist_ok=True)
        with open(plot_objects_location + filename + '.pkl', 'wb') as file:
            pickle.dump(fig, file)

    if close_plot:
        plt.close(fig)


def remove_directory(location):
    location = pathlib.Path(location)
    shutil.rmtree(location, ignore_errors=True)


def visualise_mnist(image):
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
    for feedforward_weights in feedforward_weights_list[:-1]:
        curr_values = np.matmul(curr_values, feedforward_weights)
        print("Linear: {}".format(curr_values))
        curr_values = transfer_function(curr_values)
        print("Transfer: {}".format(curr_values))

    curr_values = np.matmul(curr_values, feedforward_weights_list[-1])
    print("Linear: {}".format(curr_values))
    return curr_values


def load_model(location, name):
    pkl_file = open(location + '/' + name + '.pkl', 'rb')
    return pickle.load(pkl_file)


def save_model(save_location, name, model):
    output = open(save_location + '/' + name + '.pkl', 'wb')
    pickle.dump(model, output)


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
