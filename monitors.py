from abc import ABC, abstractmethod
import numpy as np

from helpers import create_transfer_function
import pickle

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pathlib


class Monitor(ABC):
    def __init__(self, monitor_name, var_name, plot_range=None, update_frequency=1):
        self.monitor_name = monitor_name
        self.var_name = var_name
        self.plot_range = plot_range
        self.update_frequency = update_frequency

        self.iter_numbers = []
        self.values = []

        self.plot_fig, self.plot_ax = plt.subplots()

    def plot_values(self, save_location=None):
        iter_numbers, values = self.get_values()

        print("Creating plot with {} values.".format(len(values)))
        self.plot_ax.plot(iter_numbers, values)
        self.plot_ax.set_xlim(iter_numbers[0], iter_numbers[-1])
        y_range = self.get_plot_range()
        if y_range is not None:
            self.plot_ax.set_ylim(y_range[0], y_range[1])
        self.plot_ax.set_title(self.get_var_name())
        if save_location is not None:
            filename = '/{}'.format(self.get_name())

            pathlib.Path(save_location).mkdir(parents=True, exist_ok=True)
            self.plot_fig.savefig(save_location + filename + ".pdf", bbox_inches='tight')

            plot_objects_location = save_location + '/plot_objects/'
            pathlib.Path(plot_objects_location).mkdir(parents=True, exist_ok=True)
            with open(plot_objects_location + filename + '.pkl', 'wb') as file:
                pickle.dump(self.plot_fig, file)
        self.plot_ax.clear()

    def get_name(self):
        return self.monitor_name

    def get_var_name(self):
        return self.var_name

    def get_plot_range(self):
        return self.plot_range

    def get_values(self):
        return self.iter_numbers, self.values

    def get_update_frequency(self):
        return self.update_frequency

    def prepend_data(self, prev_monitor_save_location):
        try:
            with open(prev_monitor_save_location, 'rb') as file:
                prev_monitor_iter_numbers, prev_monitor_values = pickle.load(file)
            self.iter_numbers = prev_monitor_iter_numbers + self.iter_numbers
            self.values = prev_monitor_values + self.values
            del prev_monitor_iter_numbers
            del prev_monitor_values
        except FileNotFoundError:
            print('Could not find previous monitor data at {}. Initialising from this iteration.'
                  .format(prev_monitor_save_location))

    def save_data(self, new_monitor_save_location):
        with open(new_monitor_save_location, 'wb') as file:
            monitor_iter_numbers, monitor_values = self.get_values()
            pickle.dump((monitor_iter_numbers, monitor_values), file)

    def clear_data(self):
        del self.iter_numbers
        del self.values
        self.iter_numbers = []
        self.values = []

    @abstractmethod
    def update(self, iter_number):
        pass


class GenericMonitor(Monitor):
    def __init__(self, monitor_name, var_name, monitor_function, plot_range=None, update_frequency=1):
        super().__init__(monitor_name, var_name, plot_range, update_frequency)
        self.monitor_function = monitor_function

    def update(self, iter_number):
        value = self.monitor_function(iter_number)

        self.iter_numbers += [iter_number]
        self.values += [value]


class ExponentialAverageMonitor(Monitor):
    def __init__(self, monitor, time_constant_iters):
        super().__init__(monitor.get_name(), monitor.get_var_name(), update_frequency=monitor.update_frequency)
        self.monitor = monitor
        self.decay_factor = np.exp(-1.0 / (time_constant_iters / monitor.update_frequency))

        self.averaged_values = []

    def get_averaged_values(self):
        iter_numbers, values = self.monitor.get_values()
        if len(values) == 0 or len(values) == 1:
            return iter_numbers, values

        if len(self.averaged_values) == 0:
            self.averaged_values += [values[0]]

        start_index = len(self.averaged_values)
        curr_val = self.averaged_values[start_index - 1]
        for value in values[start_index:]:
            if value is None:
                curr_val = None
            elif curr_val is None:
                curr_val = value
            else:
                curr_val = curr_val * self.decay_factor + value * (1 - self.decay_factor)
            self.averaged_values += [curr_val]
        return iter_numbers, self.averaged_values

    def get_values(self):
        return self.monitor.get_values()

    def update(self, iter_number):
        self.monitor.update(iter_number)

    def get_name(self):
        return self.monitor.get_name()

    def get_var_name(self):
        return self.monitor.get_var_name()

    def get_plot_range(self):
        return self.monitor.get_plot_range()

    def get_update_frequency(self):
        return self.monitor.get_update_frequency()

    def prepend_data(self, prev_monitor_save_location):
        self.monitor.prepend_data(prev_monitor_save_location)

    def save_data(self, new_monitor_save_location):
        self.monitor.save_data(new_monitor_save_location)

    def clear_data(self):
        self.monitor.clear_data()


class MonitorBuilder:
    @staticmethod
    def create_data_monitor(monitor_name, data_stream, data_type, data_location, update_frequency=1):
        valid_data_types = ["input", "target"]

        if data_type not in valid_data_types:
            raise Exception("Invalid data type given to monitor {}. Must be one of: {}".format(data_type,
                                                                                               valid_data_types))
        var_name = "{} location {}".format(data_type, data_location)

        if data_type == "input":
            def get_value(iter_number):
                inputs = data_stream.get_inputs(iter_number)
                return float(inputs[data_location])
        elif data_type == "target":
            def get_value(iter_number):
                output_targets = data_stream.get_output_targets(iter_number)
                if output_targets is not None:
                    return float(output_targets[data_location])
                else:
                    return None
        else:
            raise Exception("Fatal Error")

        return GenericMonitor(monitor_name, var_name, get_value, update_frequency=update_frequency)

    @staticmethod
    def create_potential_monitor(monitor_name, model, layer_num, cell_type, cell_location, update_frequency=1):
        valid_cell_types = ["pyramidal_basal", "pyramidal_soma", "pyramidal_apical", "interneuron_basal",
                            "interneuron_soma"]

        if cell_type not in valid_cell_types:
            raise Exception("Invalid cell type given to monitor {}. Must be one of: {}".format(cell_type,
                                                                                               valid_cell_types))

        var_name = "Layer {}, {} at location {}".format(layer_num, cell_type, cell_location)

        layers = model.get_layers()
        _, layer = layers[layer_num]

        if cell_type == "pyramidal_basal":
            def get_potential(iter_number):
                potentials = layer.get_pyramidal_basal_potentials()
                value = float(potentials[cell_location])
                return value
        elif cell_type == "pyramidal_soma":
            def get_potential(iter_number):
                potentials = layer.get_pyramidal_somatic_potentials()
                value = float(potentials[cell_location])
                return value
        elif cell_type == "pyramidal_apical":
            def get_potential(iter_number):
                potentials = layer.get_pyramidal_apical_potentials()
                value = float(potentials[cell_location])
                return value
        elif cell_type == "interneuron_basal":
            def get_potential(iter_number):
                potentials = layer.get_interneuron_basal_potentials()
                value = float(potentials[cell_location])
                return value
        elif cell_type == "interneuron_soma":
            def get_potential(iter_number):
                potentials = layer.get_interneuron_somatic_potentials()
                value = float(potentials[cell_location])
                return value
        else:
            raise Exception("Fatal Error")

        return GenericMonitor(monitor_name, var_name, get_potential, update_frequency=update_frequency)

    @staticmethod
    def create_weight_monitor(monitor_name, model, weight_type, layer_num, from_cell_location, to_cell_location, update_frequency=1):
        valid_weight_types = ["feedforward_weights", "predict_weights", "interneuron_weights",
                              "feedback_weights"]

        if weight_type not in valid_weight_types:
            raise Exception("Invalid weight type given to monitor {}. Must be one of: {}".format(weight_type,
                                                                                                 valid_weight_types))

        var_name = "Layer {}, {} at location {}".format(layer_num, weight_type, (from_cell_location, to_cell_location))

        layers = model.get_layers()
        _, layer = layers[layer_num]

        if weight_type == "feedforward_weights":
            weights = layer.get_feedforward_weights()
        elif weight_type == "predict_weights":
            weights = layer.get_predict_weights()
        elif weight_type == "interneuron_weights":
            weights = layer.get_interneuron_weights()
        elif weight_type == "feedback_weights":
            weights = layer.get_feedback_weights()
        else:
            raise Exception("Fatal Error")

        def get_weight(iter_number):
            value = float(weights[from_cell_location, to_cell_location])
            return value

        return GenericMonitor(monitor_name, var_name, get_weight, update_frequency=update_frequency)

    @staticmethod
    def create_weight_layer_monitor(monitor_name, model, weight_type, layer_num, operation, orig_model=None,
                                    update_frequency=1):
        valid_weight_types = ["feedforward_weights", "predict_weights", "interneuron_weights",
                              "feedback_weights"]

        if weight_type not in valid_weight_types:
            raise Exception("Invalid weight type given to monitor {}. Must be one of: {}".format(weight_type,
                                                                                                 valid_weight_types))

        layers = model.get_layers()
        _, layer = layers[layer_num]

        if weight_type == "feedforward_weights":
            weights = layer.get_feedforward_weights()
        elif weight_type == "predict_weights":
            weights = layer.get_predict_weights()
        elif weight_type == "interneuron_weights":
            weights = layer.get_interneuron_weights()
        elif weight_type == "feedback_weights":
            weights = layer.get_feedback_weights()
        else:
            raise Exception("Fatal Error")

        if orig_model is not None:
            orig_layers = orig_model.get_layers()
            _, orig_layer = orig_layers[layer_num]

            if weight_type == "feedforward_weights":
                orig_weights = orig_layer.get_feedforward_weights()
            elif weight_type == "predict_weights":
                orig_weights = orig_layer.get_predict_weights()
            elif weight_type == "interneuron_weights":
                orig_weights = orig_layer.get_interneuron_weights()
            elif weight_type == "feedback_weights":
                orig_weights = orig_layer.get_feedback_weights()
            else:
                raise Exception("Fatal Error")

        if operation == 'mean':
            def get_weight_operation_value(iter_number):
                value = float(np.mean(weights))
                return value
        elif operation == 'std':
            def get_weight_operation_value(iter_number):
                value = float(np.std(weights))
                return value
        elif operation == 'mean-magnitude':
            def get_weight_operation_value(iter_number):
                value = float(np.mean(np.abs(weights)))
                return value
        elif operation == 'std-magnitude':
            def get_weight_operation_value(iter_number):
                value = float(np.std(np.abs(weights)))
                return value
        elif operation == 'mean-magnitude-change':
            if orig_model is None:
                raise Exception("Must provide original model file to create a weight magnitude change monitor.")

            def get_weight_operation_value(iter_number):
                value = float(np.mean(np.abs(weights - orig_weights)))
                return value
        elif operation == 'std-magnitude-change':
            if orig_model is None:
                raise Exception("Must provide original model file to create a weight magnitude change monitor.")

            def get_weight_operation_value(iter_number):
                value = float(np.std(np.abs(weights - orig_weights)))
                return value
        else:
            raise Exception('Invalid operation: {}'.format(operation))

        var_name = monitor_name
        return GenericMonitor(monitor_name, var_name, get_weight_operation_value, update_frequency=update_frequency)

    @staticmethod
    def create_weight_diff_monitor(monitor_name, model, layer_num, weight_type, update_frequency=1):
        layers = model.get_layers()
        _, layer = layers[layer_num]
        _, next_layer = layers[layer_num + 1]

        if weight_type == 'feedforward_feedback_diff':
            _, next_layer = layers[layer_num + 1]

            def get_diff(num_iters):
                feedforward_weights = next_layer.get_feedforward_weights()
                feedback_weights = layer.get_feedback_weights().T
                # value = np.sum((feedforward_weights - feedback_weights) ** 2)
                value = np.linalg.norm(feedforward_weights - feedback_weights)
                return float(value)
        elif weight_type == 'feedforward_predict_diff':
            def get_diff(num_iters):
                feedforward_weights = next_layer.get_feedforward_weights()
                predict_weights = layer.get_predict_weights()
                # value = np.sum((feedforward_weights - predict_weights) ** 2)
                value = np.linalg.norm(feedforward_weights - predict_weights)
                return float(value)
        elif weight_type == 'feedback_interneuron_diff':
            def get_diff(num_iters):
                feedback_weights = layer.get_feedback_weights()
                interneuron_weights = layer.get_interneuron_weights()
                # value = np.sum((feedback_weights - interneuron_weights) ** 2)
                value = np.linalg.norm(feedback_weights - interneuron_weights)
                return float(value)
        else:
            raise Exception('Invalid weight type: {}'.format(weight_type))

        var_name = weight_type
        return GenericMonitor(monitor_name, var_name, get_diff, update_frequency=update_frequency)

    @staticmethod
    def create_weight_angle_monitor(monitor_name, model, layer_num, weight_type, update_frequency):
        layers = model.get_layers()
        _, layer = layers[layer_num]
        _, next_layer = layers[layer_num + 1]

        if weight_type == 'feedforward_feedback_angle':

            def get_angle(num_iters):
                feedforward_weights = next_layer.get_feedforward_weights()
                feedback_weights = layer.get_feedback_weights().T
                scaled_dot_product = np.dot(feedforward_weights.flatten() / np.linalg.norm(feedforward_weights),
                                            feedback_weights.flatten() / np.linalg.norm(feedback_weights))
                if np.abs(scaled_dot_product) >= 1.0:
                    angle = 0.0
                else:
                    angle = np.degrees(np.arccos(scaled_dot_product))
                return float(angle)
        elif weight_type == 'feedforward_predict_angle':
            def get_angle(num_iters):
                feedforward_weights = next_layer.get_feedforward_weights()
                predict_weights = layer.get_predict_weights()
                scaled_dot_product = np.dot(feedforward_weights.flatten() / np.linalg.norm(feedforward_weights),
                                            predict_weights.flatten() / np.linalg.norm(predict_weights))
                if np.abs(scaled_dot_product) >= 1.0:
                    angle = 0.0
                else:
                    angle = np.degrees(np.arccos(scaled_dot_product))
                return float(angle)
        elif weight_type == 'feedback_interneuron_angle':
            def get_angle(num_iters):
                feedback_weights = layer.get_feedback_weights()
                interneuron_weights = layer.get_interneuron_weights()
                scaled_dot_product = np.dot(feedback_weights.flatten() / np.linalg.norm(feedback_weights),
                                            interneuron_weights.flatten() / np.linalg.norm(interneuron_weights))
                if np.abs(scaled_dot_product) >= 1.0:
                    angle = 0.0
                else:
                    angle = np.degrees(np.arccos(scaled_dot_product))
                return float(angle)
        else:
            raise Exception('Invalid weight type: {}'.format(weight_type))

        var_name = weight_type
        return GenericMonitor(monitor_name, var_name, get_angle, plot_range=(0, 180), update_frequency=update_frequency)


    @staticmethod
    def create_pyramidal_basal_soma_rate_diff_monitor(monitor_name, model, layer_num, cell_location, dynamics_parameters, update_frequency):
        layers = model.get_layers()
        _, layer = layers[layer_num]

        leak_conductance = dynamics_parameters['leak_conductance']
        apical_conductance = dynamics_parameters['apical_conductance']
        basal_conductance = dynamics_parameters['basal_conductance']
        transfer_function = create_transfer_function(dynamics_parameters['transfer_function'])

        scaling_factor = basal_conductance / (leak_conductance + basal_conductance + apical_conductance)

        def get_pyramidal_basal_soma_rate_diff(iter_number):
            pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
            pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()

            soma_rate = transfer_function(pyramidal_somatic_potentials[cell_location, None])
            basal_rate = transfer_function(scaling_factor * pyramidal_basal_potentials[cell_location, None])

            value = soma_rate - basal_rate
            return float(value)

        var_name = 'pyramidal_basal_soma_rate_diff'
        return GenericMonitor(var_name, get_pyramidal_basal_soma_rate_diff, update_frequency=update_frequency)

    @staticmethod
    def create_error_monitor(monitor_name, model, input_output_stream, error_type, dynamics_parameters, update_frequency):
        _, last_layer = model.get_layers()[-1]

        transfer_function = create_transfer_function(dynamics_parameters['transfer_function'])

        if error_type == 'sum_squares_error':
            def get_error(num_iters):
                target = input_output_stream.get_output_targets(num_iters)
                if target is None:
                    return None
                else:
                    target_rate = transfer_function(target)
                    output_rate = transfer_function(last_layer.get_pyramidal_somatic_potentials())
                    error = np.sum((target_rate - output_rate)**2)
                    return float(error)
        else:
            raise Exception('Invalid error type {}'.format(error_type))

        var_name = error_type
        return GenericMonitor(monitor_name, var_name, get_error, update_frequency=update_frequency)