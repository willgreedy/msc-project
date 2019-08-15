import matplotlib
import pickle
import numpy as np


def set_params():
    params = {
        'axes.labelsize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    }
    matplotlib.rcParams.update(params)


def load_object(location):
    with open(location, 'rb') as file:
        plot_object = pickle.load(file)
    return plot_object


def append_data(data_tuple_list):
    iter_data = []
    val_data = []
    for iter, val in data_tuple_list:
        iter_data += iter
        val_data += val

    return iter_data, val_data


def load_append_data(base_path, num_objects):
    objects = []
    for i in range(num_objects):
        object_path = base_path + '_' + str(i+1) + '.pkl'
        objects += [load_object(object_path)]
    return append_data(objects)


def get_plot_data(plot_object):
    ax = plot_object.axes[0]
    line = ax.lines[0]
    data = line.get_xydata()
    return data[:, 0], data[:, 1]


def get_smoothed_data(raw_data, time_constant):
    decay_constant = np.exp(-1 / time_constant)
    averaged_values = []
    starting_value = np.mean(raw_data[:1000])
    curr_val = starting_value
    for value in raw_data:
        if value is None:
            curr_val = None
        elif curr_val is None:
            curr_val = value
        else:
            curr_val = curr_val * decay_constant + value * (1 - decay_constant)
        averaged_values += [curr_val]
    return averaged_values
