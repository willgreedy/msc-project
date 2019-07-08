from abc import ABC, abstractmethod
import numpy as np

from helpers import create_transfer_function


class Monitor(ABC):
    def __init__(self, var_name, plot_range=None, update_frequency=1):
        self.var_name = var_name
        self.plot_range = plot_range
        self.update_frequency = update_frequency

        self.iter_numbers = []
        self.values = []

    def get_var_name(self):
        return self.var_name

    def get_plot_range(self):
        return self.plot_range

    def get_values(self):
        return self.iter_numbers, self.values

    def get_update_frequency(self):
        return self.update_frequency

    @abstractmethod
    def update(self, iter_number):
        pass


class GenericMonitor(Monitor):
    def __init__(self, var_name, monitor_function, plot_range=None, update_frequency=1):
        super().__init__(var_name, plot_range, update_frequency)
        self.monitor_function = monitor_function

    def update(self, iter_number):
        value = self.monitor_function(iter_number)

        self.iter_numbers += [iter_number]
        self.values += [value]


class ExponentialAverageMonitor(Monitor):
    def __init__(self, monitor, time_constant_iters):
        super().__init__(monitor.var_name, update_frequency=monitor.update_frequency)
        self.monitor = monitor
        self.decay_factor = np.exp(-1.0 / (time_constant_iters / monitor.update_frequency))

    def get_values(self):
        iter_numbers, values = self.monitor.get_values()
        if len(values) == 0 or len(values) == 1:
            return iter_numbers, values

        averaged_values = [values[0]]
        curr_val = values[0]
        for value in values[1:]:
            if value is None:
                curr_val = None
            elif curr_val is None:
                curr_val = value
            else:
                curr_val = curr_val * self.decay_factor + value * (1 - self.decay_factor)
            averaged_values += [curr_val]
        return iter_numbers, averaged_values

    def update(self, iter_number):
        self.monitor.update(iter_number)


class MonitorBuilder:
    @staticmethod
    def create_data_monitor(data_stream, data_type, data_location, update_frequency=1):
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

        return GenericMonitor(var_name, get_value, update_frequency=update_frequency)

    @staticmethod
    def create_potential_monitor(model, layer_num, cell_type, cell_location, update_frequency=1):
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

        return GenericMonitor(var_name, get_potential, update_frequency=update_frequency)

    @staticmethod
    def create_weight_monitor(model, weight_type, layer_num, from_cell_location, to_cell_location, update_frequency=1):
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

        return GenericMonitor(var_name, get_weight, update_frequency=update_frequency)

    @staticmethod
    def create_feedforward_predict_weight_diff_monitor(model, feedforward_layer_num, predict_layer_num, update_frequency=1):
        layers = model.get_layers()
        _, feedforward_layer = layers[feedforward_layer_num]
        _, predict_layer = layers[predict_layer_num]

        def get_feedforward_predict_weights_diff(iter_number):
            feedforward_weights = feedforward_layer.get_feedforward_weights()
            predict_weights = predict_layer.get_predict_weights()
            value = np.sum((feedforward_weights - predict_weights) ** 2)

            return float(value)

        var_name = 'feedforward_predict_weight_diff'
        return GenericMonitor(var_name, get_feedforward_predict_weights_diff, update_frequency=update_frequency)

    @staticmethod
    def create_pyramidal_basal_soma_rate_diff_monitor(model, layer_num, cell_location, dynamics_parameters, update_frequency):
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
    def create_weight_angle_monitor(model, layer_num, weight_type, update_frequency):
        layers = model.get_layers()
        _, layer = layers[layer_num]
        _, next_layer = layers[layer_num + 1]

        if weight_type == 'feedforward_feedback_angle':
            _, next_layer = layers[layer_num + 1]
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
        return GenericMonitor(var_name, get_angle, plot_range=(0, 180), update_frequency=update_frequency)

    @staticmethod
    def create_error_monitor(model, input_output_stream, error_type, dynamics_parameters, update_frequency):
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
        return GenericMonitor(var_name, get_error, update_frequency=update_frequency)