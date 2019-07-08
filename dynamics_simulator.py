from abc import ABC, abstractmethod

import numpy as np
from helpers import create_transfer_function, visualise_transfer_function, save_model
from layers import StandardLayer, OutputPyramidalLayer


class DynamicsSimulator(ABC):
    def __init__(self, model, input_output_stream, dynamics_parameters, monitors=list()):
        self.model = model
        self.input_output_stream = input_output_stream
        self.iter_step = 0
        self.dynamics_parameters = dynamics_parameters
        self.background_noise_std = dynamics_parameters['background_noise_std']
        self.transfer_function = create_transfer_function(dynamics_parameters['transfer_function'])
        # visualise_transfer_function(self.transfer_function)

        self.plastic_feedforward_weights = dynamics_parameters['plastic_feedforward_weights']
        self.plastic_predict_weights = dynamics_parameters['plastic_predict_weights']
        self.plastic_interneuron_weights = dynamics_parameters['plastic_interneuron_weights']
        self.plastic_feedback_weights = dynamics_parameters['plastic_feedback_weights']

        self.monitors = monitors
        self.testing_phase = False

    def run_simulation(self, max_iterations):
        report_interval = int((max_iterations - self.iter_step) / 10)

        while self.iter_step < max_iterations:
            self.iter_step += 1
            if self.iter_step % report_interval == 0:
                print("Iteration {}.".format(self.iter_step))

            self.step_simulation()
            for monitor in self.monitors:
                if self.iter_step % monitor.get_update_frequency() == 0:
                    monitor.update(self.iter_step)

    def set_testing_phase(self, testing_phase):
        self.testing_phase = testing_phase

    def save_model(self, name):
        save_model(name, self.model)

    @abstractmethod
    def step_simulation(self):
        pass


class StandardDynamicsSimulator(DynamicsSimulator):
    def __init__(self, model, input_output_stream, dynamics_parameters, monitors=list()):
        super().__init__(model, input_output_stream, dynamics_parameters, monitors)

        self.resting_potential = dynamics_parameters['resting_potential']
        self.leak_conductance = dynamics_parameters['leak_conductance']
        self.apical_conductance = dynamics_parameters['apical_conductance']
        self.basal_conductance = dynamics_parameters['basal_conductance']
        self.dendritic_conductance = dynamics_parameters['dendritic_conductance']
        self.nudging_conductance = dynamics_parameters['nudging_conductance']

        self.step_size = dynamics_parameters['ms_per_time_step']
        if dynamics_parameters['weight_time_constant_ms'] == 0:
            self.weight_update_factor = 0.0
        else:
            self.weight_update_factor = np.exp(
                -1.0 / (dynamics_parameters['weight_time_constant_ms'] / float(dynamics_parameters['ms_per_time_step'])))

    def step_simulation(self):
        self.compute_updates()
        self.perform_updates()

    def compute_updates(self):
        inputs = self.input_output_stream.get_inputs(self.iter_step)
        output_targets = self.input_output_stream.get_output_targets(self.iter_step)

        layers = self.model.get_layers()

        for layer_index, (_, layer) in enumerate(layers):
            if isinstance(layer, StandardLayer):
                if layer_index == 0:
                    prev_layer_pyramidal_somatic_potentials = inputs
                elif layer_index < len(layers) - 1:
                    _, prev_layer = layers[layer_index - 1]
                    prev_layer_pyramidal_somatic_potentials = prev_layer.get_pyramidal_somatic_potentials()
                else:
                    raise Exception('Invalid layer specification! StandardLayer cannot be used as the final layer.')

                _, next_layer = layers[layer_index + 1]
                next_layer_pyramidal_somatic_potentials = next_layer.get_pyramidal_somatic_potentials()

                self.compute_standard_layer_updates(layer,
                                                    prev_layer_pyramidal_somatic_potentials,
                                                    next_layer_pyramidal_somatic_potentials)

            elif isinstance(layer, OutputPyramidalLayer):
                if layer_index > 0:
                    _, prev_layer = layers[layer_index - 1]
                    prev_layer_pyramidal_somatic_potentials = prev_layer.get_pyramidal_somatic_potentials()
                    self.compute_output_layer_updates(layer, prev_layer_pyramidal_somatic_potentials, output_targets)
                else:
                    raise Exception('Invalid layer specification! OutputPyramidalLayer cannot be used as the first layer.')

    def perform_updates(self):
        layers = self.model.get_layers()
        for layer_index, (_, layer) in enumerate(layers):
            layer.perform_update(self.step_size, self.weight_update_factor)

    def compute_standard_layer_updates(self, layer, prev_layer_pyramidal_somatic_potentials,
                                       next_layer_pyramidal_somatic_potentials):

        # Compute somatic compartment updates
        change_pyramidal_somatic_potentials = self.compute_standard_pyramidal_somatic_potential_updates(layer)
        change_interneuron_somatic_potentials = self.compute_interneuron_somatic_potential_updates(layer,
                                                                                                   next_layer_pyramidal_somatic_potentials)

        # Compute basal and apical compartment updates
        new_pyramidal_basal_potentials = self.compute_pyramidal_basal_potential_updates(layer,
                                                                                        prev_layer_pyramidal_somatic_potentials)

        new_pyramidal_apical_potentials = self.compute_pyramidal_apical_potential_updates(layer,
                                                                                         next_layer_pyramidal_somatic_potentials)
        new_interneuron_basal_potentials = self.compute_interneuron_basal_potential_updates(layer)

        # Compute changes in weights
        if self.plastic_feedforward_weights and not self.testing_phase:
            change_feedforward_weights = self.compute_feedforward_weight_updates(layer,
                                                                             prev_layer_pyramidal_somatic_potentials)
        else:
            change_feedforward_weights = np.zeros(layer.get_feedforward_weights().shape)

        if self.plastic_predict_weights and not self.testing_phase:
            change_predict_weights = self.compute_predict_weight_updates(layer)
        else:
            change_predict_weights = np.zeros(layer.get_predict_weights().shape)

        if self.plastic_interneuron_weights and not self.testing_phase:
            change_interneuron_weights = self.compute_interneuron_weight_updates(layer)
        else:
            change_interneuron_weights = np.zeros(layer.get_interneuron_weights().shape)

        if self.plastic_feedback_weights and not self.testing_phase:
            change_feedback_weights = self.compute_feedback_weight_updates(layer,
                                                                           next_layer_pyramidal_somatic_potentials)
        else:
            change_feedback_weights = np.zeros(layer.get_feedback_weights().shape)

        layer.set_change_pyramidal_somatic_potentials(change_pyramidal_somatic_potentials)
        layer.set_change_interneuron_somatic_potentials(change_interneuron_somatic_potentials)
        layer.set_new_pyramidal_basal_potentials(new_pyramidal_basal_potentials)
        layer.set_new_pyramidal_apical_potentials(new_pyramidal_apical_potentials)
        layer.set_new_interneuron_basal_potentials(new_interneuron_basal_potentials)
        layer.set_change_feedforward_weights(change_feedforward_weights, self.weight_update_factor)
        layer.set_change_predict_weights(change_predict_weights, self.weight_update_factor)
        layer.set_change_interneuron_weights(change_interneuron_weights, self.weight_update_factor)
        layer.set_change_feedback_weights(change_feedback_weights, self.weight_update_factor)

    def compute_output_layer_updates(self, layer, prev_layer_pyramidal_somatic_potentials, output_targets):
        change_pyramidal_somatic_potentials = self.compute_output_pyramidal_somatic_potential_updates(layer,
                                                                                                      output_targets)

        new_pyramidal_basal_potentials = self.compute_pyramidal_basal_potential_updates(layer,
                                                                                        prev_layer_pyramidal_somatic_potentials)

        if self.plastic_feedforward_weights and not self.testing_phase:
            change_feedforward_weights = self.compute_output_layer_feedforward_weight_updates(layer,
                                                                             prev_layer_pyramidal_somatic_potentials)
        else:
            change_feedforward_weights = np.zeros(layer.get_feedforward_weights().shape)

        layer.set_change_pyramidal_somatic_potentials(change_pyramidal_somatic_potentials)
        layer.set_new_pyramidal_basal_potentials(new_pyramidal_basal_potentials)
        layer.set_change_feedforward_weights(change_feedforward_weights, self.weight_update_factor)

    def compute_standard_pyramidal_somatic_potential_updates(self, layer):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()
        pyramidal_apical_potentials = layer.get_pyramidal_apical_potentials()

        # Compute somatic compartment updates
        if self.testing_phase:
            background_noise = np.zeros((layer.num_neurons, 1))
        else:
            background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                            size=(layer.num_neurons, 1))

        change_pyramidal_somatic_potentials = -self.leak_conductance * pyramidal_somatic_potentials + \
                                              self.basal_conductance * (pyramidal_basal_potentials -
                                                                        pyramidal_somatic_potentials) +\
                                              self.apical_conductance * (pyramidal_apical_potentials -
                                                                         pyramidal_somatic_potentials) +\
                                              background_noise

        return change_pyramidal_somatic_potentials

    def compute_output_pyramidal_somatic_potential_updates(self, layer, output_targets):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()

        # Compute somatic compartment updates
        if self.testing_phase:
            background_noise = np.zeros((layer.num_neurons, 1))
        else:
            background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                            size=(layer.num_neurons, 1))

        if output_targets is not None and not self.testing_phase:
            change_pyramidal_somatic_potentials = -self.leak_conductance * pyramidal_somatic_potentials + \
                                              self.basal_conductance * (pyramidal_basal_potentials -
                                                                        pyramidal_somatic_potentials) +\
                                              self.nudging_conductance * (output_targets -
                                                                          pyramidal_somatic_potentials) +\
                                              background_noise
        else:
            change_pyramidal_somatic_potentials = -self.leak_conductance * pyramidal_somatic_potentials + \
                                                  self.basal_conductance * (pyramidal_basal_potentials -
                                                                            pyramidal_somatic_potentials) + \
                                                  background_noise

        #print("Target: {}, Soma {}, Diff {}".format(output_targets[0, 0],
        #                                            pyramidal_somatic_potentials[0,0],
        #                                            output_targets[0, 0] - pyramidal_somatic_potentials[0, 0]))
        #print("Basal {}, Soma {}, Diff {}".format(pyramidal_basal_potentials[0,0],
        #                                          pyramidal_somatic_potentials[0, 0],
        #                                          pyramidal_basal_potentials[0,0] - pyramidal_somatic_potentials[0,0]))
        #print("Change {}".format(change_pyramidal_somatic_potentials[0,0]))
        return change_pyramidal_somatic_potentials

    def compute_interneuron_somatic_potential_updates(self, layer, next_layer_pyramidal_somatic_potentials):
        interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()
        interneuron_basal_potentials = layer.get_interneuron_basal_potentials()

        #teaching_feedback = np.zeros((layer.num_neurons_next, 1))
        teaching_feedback = self.nudging_conductance * (next_layer_pyramidal_somatic_potentials -
                                                        interneuron_somatic_potentials)

        if self.testing_phase:
            background_noise = np.zeros((layer.num_neurons_next, 1))
        else:
            background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                            size=(layer.num_neurons_next, 1))

        change_interneuron_somatic_potentials = -self.leak_conductance * interneuron_somatic_potentials + \
                                                self.dendritic_conductance * (
                                                    interneuron_basal_potentials - interneuron_somatic_potentials) + \
                                                teaching_feedback + background_noise

        return change_interneuron_somatic_potentials

    def compute_pyramidal_basal_potential_updates(self, layer, prev_layer_pyramidal_somatic_potentials):
        feedforward_weights = layer.get_feedforward_weights()

        prev_pyramidal_firing_rates = self.transfer_function(prev_layer_pyramidal_somatic_potentials)
        new_pyramidal_basal_potentials = np.dot(feedforward_weights, prev_pyramidal_firing_rates)

        return new_pyramidal_basal_potentials

    def compute_pyramidal_apical_potential_updates(self, layer, next_layer_pyramidal_somatic_potentials):
        interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()

        feedback_weights = layer.get_feedback_weights()
        interneuron_weights = layer.get_interneuron_weights()

        next_pyramidal_firing_rates = self.transfer_function(next_layer_pyramidal_somatic_potentials)
        new_pyramidal_apical_potentials = np.dot(feedback_weights, next_pyramidal_firing_rates)

        interneuron_firing_rates = self.transfer_function(interneuron_somatic_potentials)
        new_pyramidal_apical_potentials += np.dot(interneuron_weights, interneuron_firing_rates)

        #print(np.dot(feedback_weights, next_pyramidal_firing_rates))
        #print(np.dot(interneuron_weights, interneuron_firing_rates).shape)

        return new_pyramidal_apical_potentials

    def compute_interneuron_basal_potential_updates(self, layer):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        predict_weights = layer.get_predict_weights()

        new_interneuron_basal_potential = np.dot(predict_weights, self.transfer_function(pyramidal_somatic_potentials))
        return new_interneuron_basal_potential

    def compute_feedforward_weight_updates(self, layer, prev_layer_pyramidal_somatic_potentials):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()

        feedforward_learning_rate = layer.get_feedforward_learning_rate()

        scaling_factor = self.basal_conductance / (self.leak_conductance + self.basal_conductance +
                                                             self.apical_conductance)

        prev_pyramidal_firing_rates = self.transfer_function(prev_layer_pyramidal_somatic_potentials)

        change_feedforward_weights = feedforward_learning_rate * \
                                     np.dot((self.transfer_function(pyramidal_somatic_potentials) -
                                             self.transfer_function(scaling_factor *
                                                                    pyramidal_basal_potentials)),
                                            prev_pyramidal_firing_rates.T)

        #print("Max weight: {}".format(np.max(layer.get_feedforward_weights())))
        #print("Max value: {}".format(np.max((self.transfer_function(pyramidal_somatic_potentials) -
        #                                     self.transfer_function(scaling_factor *
        #                                                            pyramidal_basal_potentials)))))
        #print("Max change: {}".format(np.max(change_feedforward_weights)))
        #print("Input Firing Rate: {}".format(scaling_factor * pyramidal_basal_potentials[0,0]))
        return change_feedforward_weights

    def compute_output_layer_feedforward_weight_updates(self, layer, prev_layer_pyramidal_somatic_potentials):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()

        feedforward_learning_rate = layer.get_feedforward_learning_rate()

        scaling_factor = self.basal_conductance / (self.leak_conductance + self.basal_conductance)

        prev_pyramidal_firing_rates = self.transfer_function(prev_layer_pyramidal_somatic_potentials)

        change_feedforward_weights = feedforward_learning_rate * \
                                     np.dot((self.transfer_function(pyramidal_somatic_potentials) -
                                             self.transfer_function(scaling_factor *
                                                                    pyramidal_basal_potentials)),
                                            prev_pyramidal_firing_rates.T)

        return change_feedforward_weights

    def compute_predict_weight_updates(self, layer):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()

        interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()
        interneuron_basal_potentials = layer.get_interneuron_basal_potentials()

        predict_learning_rate = layer.get_predict_learning_rate()

        scaling_factor = self.dendritic_conductance / (self.leak_conductance +
                                                                   self.dendritic_conductance)

        pyramidal_firing_rates = self.transfer_function(pyramidal_somatic_potentials)

        change_predict_weights = predict_learning_rate * \
                                 np.dot((self.transfer_function(interneuron_somatic_potentials) -
                                         self.transfer_function(scaling_factor *
                                                                interneuron_basal_potentials)),
                                        pyramidal_firing_rates.T)

        return change_predict_weights

    def compute_interneuron_weight_updates(self, layer):
        pyramidal_apical_potentials = layer.get_pyramidal_apical_potentials()

        interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()

        interneuron_learning_rate = layer.get_interneuron_learning_rate()

        interneuron_firing_rates = self.transfer_function(interneuron_somatic_potentials)

        change_interneuron_weights = interneuron_learning_rate * \
                                     np.dot((self.resting_potential - pyramidal_apical_potentials),
                                            interneuron_firing_rates.T)

        return change_interneuron_weights

    def compute_feedback_weight_updates(self, layer, next_layer_pyramidal_somatic_potentials):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        feedback_weights = layer.get_feedback_weights()

        feedback_learning_rate = layer.get_feedback_learning_rate()

        next_pyramidal_firing_rates = self.transfer_function(next_layer_pyramidal_somatic_potentials)

        top_down_inputs = np.dot(feedback_weights, next_pyramidal_firing_rates)
        change_feedback_weights = feedback_learning_rate * \
                                      np.dot((self.transfer_function(pyramidal_somatic_potentials) -
                                              self.transfer_function(top_down_inputs)),
                                             next_pyramidal_firing_rates.T)
        return change_feedback_weights


class SimplifiedDynamicsSimulator(DynamicsSimulator):
    def __init__(self, model, input_output_stream, dynamics_parameters, monitors=list()):
        super().__init__(model, input_output_stream, dynamics_parameters, monitors)

        self.somatic_mixing_factors = dynamics_parameters['somatic_mixing_factors']
        self.interneuron_mixing_factors = dynamics_parameters['interneuron_mixing_factors']

    def step_simulation(self):
        self.perform_updates()

    def perform_updates(self):
        inputs = self.input_output_stream.get_inputs(0)
        output_targets = self.input_output_stream.get_output_targets(0)
        layers = self.model.get_layers()

        for layer_index, (_, layer) in enumerate(layers):
            if isinstance(layer, StandardLayer):
                if layer_index == 0:
                    prev_layer_pyramidal_somatic_potentials = inputs
                elif layer_index < len(layers) - 1:
                    _, prev_layer = layers[layer_index - 1]
                    prev_layer_pyramidal_somatic_potentials = prev_layer.get_pyramidal_somatic_potentials()
                else:
                    raise Exception('Invalid layer specification! StandardLayer cannot be used as the final layer.')

                pyramidal_basal_potentials = self.compute_pyramidal_basal_potential_updates(layer, inputs, prev_layer_pyramidal_somatic_potentials)
                pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
                pyramidal_somatic_potentials_change = pyramidal_basal_potentials - pyramidal_somatic_potentials
                layer.set_new_pyramidal_basal_potentials(pyramidal_basal_potentials)
                layer.set_change_pyramidal_somatic_potentials(pyramidal_somatic_potentials_change)
            elif isinstance(layer, OutputPyramidalLayer):
                _, prev_layer = layers[layer_index - 1]
                prev_layer_pyramidal_somatic_potentials = prev_layer.get_pyramidal_somatic_potentials()
                self.compute_output_pyramidal_somatic_potential_updates(layer, prev_layer_pyramidal_somatic_potentials, output_targets)
            else:
                raise Exception('Invalid layer specification! OutputPyramidalLayer cannot be used as the first layer.')

        # Visit reverse order from k
        for i, (_, layer) in enumerate(layers[:-1:-1]):
            layer_index = len(layers) - 1 - i
            print("Layer index: {}".format(layer_index))
            if isinstance(layer, StandardLayer):
                _, next_layer = layers[layer_index + 1]
                if layer_index < len(layers) - 1:
                    next_layer_pyramidal_somatic_potentials = next_layer.get_pyramidal_somatic_potentials()
                    self.compute_standard_pyramidal_somatic_potential_updates(layer, self.somatic_mixing_factors[layer_index])
                    self.compute_interneuron_basal_potential_updates(layer)
                    self.compute_interneuron_somatic_potential_updates(layer, next_layer_pyramidal_somatic_potentials)
                    self.compute_pyramidal_apical_potential_updates(layer, next_layer_pyramidal_somatic_potentials)
                else:
                    raise Exception('Invalid layer specification! StandardLayer cannot be used as the final layer.')
            elif isinstance(layer, OutputPyramidalLayer):
                raise Exception(
                    'Invalid layer specification! OutputPyramidalLayer cannot be used as anything other than the last layer.')

    def compute_pyramidal_basal_potential_updates(self, layer, prev_layer_pyramidal_somatic_potentials):
        feedforward_weights = layer.get_feedforward_weights()

        prev_pyramidal_firing_rates = self.transfer_function(prev_layer_pyramidal_somatic_potentials)
        new_pyramidal_basal_potentials = np.dot(feedforward_weights, prev_pyramidal_firing_rates)

        return new_pyramidal_basal_potentials

    def compute_standard_pyramidal_somatic_potential_updates(self, layer, mixing_factor):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()
        pyramidal_apical_potentials = layer.get_pyramidal_apical_potentials()

    def compute_output_pyramidal_somatic_potential_updates(self, layer):
        pass

    def compute_pyramidal_apical_potential_updates(self, layer, next_layer_pyramidal_somatic_potentials):
        pass

    def compute_interneuron_basal_potential_updates(self, layer):
        pass

    def compute_interneuron_somatic_potential_updates(self, layer, next_layer_pyramidal_somatic_potentials):
        pass





