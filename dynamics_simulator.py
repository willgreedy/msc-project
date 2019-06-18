import numpy as np
from helpers import create_transfer_function
from layers import StandardLayer, OutputPyramidalLayer


class DynamicsSimulator:
    def __init__(self, model, parameters, monitors=list()):
        self.model = model
        self.iter_step = 0

        self.parameters = parameters
        self.resting_potential = self.parameters['resting_potential']
        self.leak_conductance = self.parameters['leak_conductance']
        self.apical_conductance = self.parameters['apical_conductance']
        self.basal_conductance = self.parameters['basal_conductance']
        self.dendritic_conductance = self.parameters['dendritic_conductance']
        self.nudging_conductance = self.parameters['nudging_conductance']
        self.background_noise_std = self.parameters['background_noise_std']
        self.transfer_function = create_transfer_function(self.parameters['transfer_function'])
        self.plastic_feedback_weights = self.parameters['plastic_feedback_weights']
        self.monitors = monitors

    def run_simulation(self, max_iterations):
        for monitor in self.monitors:
            monitor.update(self.iter_step)

        while self.iter_step < max_iterations:
            self.iter_step += 1
            self.step_simulation()
            for monitor in self.monitors:
                monitor.update(self.iter_step)

    def step_simulation(self):
        self.compute_updates()
        self.perform_updates()
        self.reset_stored_updates()

    def compute_updates(self):
        # inputs = self.data_stream.get_inputs(self.iter_step)
        #inputs = np.zeros((self.model.input_size, 1))
        inputs = np.ones((self.model.input_size, 1))

        # output_targets = self.data_stream.get_output_target(self.iter_step)
        output_targets = np.zeros((self.model.output_size, 1))

        layers = self.model.get_layers()

        for layer_index, (_, layer) in enumerate(layers):

            if isinstance(layer, StandardLayer):
                if layer_index == 0:
                    prev_layer_pyramidal_somatic_potentials = inputs
                elif layer_index < len(layers) - 1:
                    prev_layer = layers[layer_index - 1]
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

    def compute_standard_layer_updates(self, layer, prev_layer_pyramidal_somatic_potentials,
                                       next_layer_pyramidal_somatic_potentials):

        # Compute somatic compartment updates
        change_pyramidal_somatic_potentials = self.compute_standard_pyramidal_somatic_potential_updates(layer)
        change_interneuron_somatic_potentials = self.compute_interneuron_somatic_potential_updates(layer)

        # Compute basal and apical compartment updates
        new_pyramidal_basal_potentials = self.compute_pyramidal_basal_potential_updates(layer,
                                                                                        prev_layer_pyramidal_somatic_potentials)
        new_pyramidal_apical_potentials = self.compute_pyramidal_apical_potential_updates(layer,
                                                                                         next_layer_pyramidal_somatic_potentials)
        new_interneuron_basal_potentials = self.compute_interneuron_basal_potential_updates(layer)

        # Compute changes in weights
        change_feedforward_weights = self.compute_feedforward_weight_updates(layer,
                                                                             prev_layer_pyramidal_somatic_potentials)
        change_predict_weights = self.compute_predict_weight_updates(layer)
        change_interneuron_weights = self.compute_interneuron_weight_updates(layer)
        change_feedback_weights = self.compute_feedback_weight_updates(layer, next_layer_pyramidal_somatic_potentials)

        layer.set_change_pyramidal_somatic_potentials(change_pyramidal_somatic_potentials)
        layer.set_change_interneuron_somatic_potentials(change_interneuron_somatic_potentials)
        layer.set_new_pyramidal_basal_potentials(new_pyramidal_basal_potentials)
        layer.set_new_pyramidal_apical_potentials(new_pyramidal_apical_potentials)
        layer.set_new_interneuron_basal_potentials(new_interneuron_basal_potentials)
        layer.set_change_feedforward_weights(change_feedforward_weights)
        layer.set_change_predict_weights(change_predict_weights)
        layer.set_change_interneuron_weights(change_interneuron_weights)
        layer.set_change_feedback_weights(change_feedback_weights)

    def compute_output_layer_updates(self, layer, prev_layer_pyramidal_somatic_potentials, output_targets):
        change_pyramidal_somatic_potentials = self.compute_output_pyramidal_somatic_potential_updates(layer,
                                                                                                      output_targets)

        new_pyramidal_basal_potentials = self.compute_pyramidal_basal_potential_updates(layer,
                                                                                        prev_layer_pyramidal_somatic_potentials)

        change_feedforward_weights = self.compute_feedforward_weight_updates(layer,
                                                                             prev_layer_pyramidal_somatic_potentials)

        layer.set_change_pyramidal_somatic_potentials(change_pyramidal_somatic_potentials)
        layer.set_new_pyramidal_basal_potentials(new_pyramidal_basal_potentials)
        layer.set_change_feedforward_weights(change_feedforward_weights)

    def compute_standard_pyramidal_somatic_potential_updates(self, layer):
        pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
        pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()
        pyramidal_apical_potentials = layer.get_pyramidal_apical_potentials()

        # Compute somatic compartment updates
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
        background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                            size=(layer.num_neurons, 1))

        change_pyramidal_somatic_potentials = -self.leak_conductance * pyramidal_somatic_potentials + \
                                              self.basal_conductance * (pyramidal_basal_potentials -
                                                                        pyramidal_somatic_potentials) +\
                                              self.nudging_conductance * (output_targets -
                                                                          pyramidal_somatic_potentials) +\
                                              background_noise

        return change_pyramidal_somatic_potentials

    def compute_interneuron_somatic_potential_updates(self, layer):
        interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()
        interneuron_basal_potentials = layer.get_interneuron_basal_potentials()

        cross_layer_feedback = np.zeros((layer.num_neurons_next, 1))
        background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                            size=(layer.num_neurons_next, 1))

        change_interneuron_somatic_potentials = -self.leak_conductance * interneuron_somatic_potentials + \
                                                self.dendritic_conductance * (
                                                    interneuron_basal_potentials - interneuron_somatic_potentials) + \
                                                cross_layer_feedback + background_noise

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

        if self.plastic_feedback_weights:
            top_down_inputs = np.dot(feedback_weights, next_pyramidal_firing_rates)
            change_feedback_weights = feedback_learning_rate * \
                                      np.dot((self.transfer_function(pyramidal_somatic_potentials) -
                                              self.transfer_function(top_down_inputs)),
                                             next_pyramidal_firing_rates.T)
        else:
            change_feedback_weights = None

        return change_feedback_weights

    def perform_updates(self):
        # TODO: Fix step_size
        step_size = 0.1
        layers = self.model.get_layers()

        for layer_index, (_, layer) in enumerate(layers):
            layer.perform_update(step_size)

    def reset_stored_updates(self):
        layers = self.model.get_layers()

        for layer_index, (_, layer) in enumerate(layers):
            layer.reset_stored_updates()

    '''def compute_updates(self):
        # inputs = self.data_stream.get_inputs(self.iter_step)
        inputs = np.zeros((self.model.input_size,))

        # output = self.data_stream.get_output_target(self.iter_step)
        output_target = np.zeros((self.model.output_size,))

        layers = self.model.get_layers()

        # Compute changes to neuron potentials and synaptic weights
        for layer_index in range(len(layers) - 1):

            layer = layers[layer_index]

            pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
            pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()
            pyramidal_apical_potentials = layer.get_pyramidal_apical_potentials()

            interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()
            interneuron_basal_potentials = layer.get_interneuron_basal_potentials()

            feedback_weights = layer.get_feedback_weights()
            feedforward_weights = layer.get_feedforward_weights()
            interneuron_weights = layer.get_interneuron_weights()
            predict_weights = layer.get_predict_weights()

            # Compute somatic compartment updates
            background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                                size=pyramidal_somatic_potentials.shape)

            change_pyramidal_somatic_potentials = -self.leak_conductance * pyramidal_somatic_potentials + \
                self.basal_conductance * (pyramidal_basal_potentials - pyramidal_somatic_potentials) + \
                self.apical_conductance * (pyramidal_apical_potentials - pyramidal_somatic_potentials) + \
                background_noise

            cross_layer_feedback = np.zeros(interneuron_somatic_potentials.shape)
            background_noise = np.random.normal(loc=0, scale=self.background_noise_std,
                                                size=interneuron_somatic_potentials.shape)

            change_interneuron_somatic_potentials = -self.leak_conductance * interneuron_somatic_potentials + \
                self.dendritic_conductance * (interneuron_basal_potentials - interneuron_somatic_potentials) + \
                cross_layer_feedback + background_noise

            # Compute pyramidal basal compartment updates
            if layer_index > 0:
                # Compute for hidden layer
                prev_layer = layers[layer_index - 1]
                prev_layer_pyramidal_somatic_potentials = prev_layer.get_pyramidal_somatic_potentials()
                prev_pyramidal_firing_rates = self.transfer_function(prev_layer_pyramidal_somatic_potentials)
            else:
                # Compute for first layer
                prev_pyramidal_firing_rates = self.transfer_function(inputs)

            new_pyramidal_basal_potentials = np.dot(feedforward_weights, prev_pyramidal_firing_rates)

            # Compute pyramidal apical compartment updates
            if layer_index < len(layers) - 2:
                # Compute for all but last layer
                next_layer = layers[layer_index + 1]
                next_layer_pyramidal_somatic_potentials = next_layer.get_pyramidal_somatic_potentials()
                next_pyramidal_firing_rates = self.transfer_function(
                                                                next_layer_pyramidal_somatic_potentials)
            else:
                # Compute for last layer
                next_pyramidal_firing_rates = self.transfer_function(output_target)

            new_pyramidal_apical_potentials = np.dot(feedback_weights, next_pyramidal_firing_rates)

            interneuron_firing_rates = self.transfer_function(interneuron_somatic_potentials)
            new_pyramidal_apical_potentials += np.dot(interneuron_weights, interneuron_firing_rates)

            # Compute interneuron basal potential
            new_interneuron_basal_potential = np.dot(predict_weights,
                                                     self.transfer_function(pyramidal_somatic_potentials))

            # Get layer learning rates
            feedforward_learning_rate = layer.get_feedforward_learning_rate()
            predict_learning_rate = layer.get_predict_learning_rate()
            interneuron_learning_rate = layer.get_interneuron_learning_rate()
            feedback_learning_rate = layer.get_feedback_learning_rate()

            pyramidal_firing_rates = self.transfer_function(pyramidal_somatic_potentials)

            pyramidal_scaling_factor = self.basal_conductance / (self.leak_conductance + self.basal_conductance +
                                                                 self.apical_conductance)

            interneuron_scaling_factor = self.dendritic_conductance / (self.leak_conductance +
                                                                       self.dendritic_conductance)

            # Compute change in weights
            change_feedforward_weights = feedback_learning_rate *\
                                         np.dot((self.transfer_function(pyramidal_somatic_potentials) -
                                                 self.transfer_function(pyramidal_scaling_factor *
                                                                        pyramidal_basal_potentials)),
                                                prev_pyramidal_firing_rates.T)

            change_predict_weights = predict_learning_rate *\
                                     np.dot((self.transfer_function(interneuron_somatic_potentials) -
                                             self.transfer_function(interneuron_scaling_factor *
                                                                    interneuron_basal_potentials)),
                                            pyramidal_firing_rates.T)

            change_interneuron_weights = interneuron_learning_rate *\
                                         np.dot((self.resting_potential - pyramidal_apical_potentials),
                                                interneuron_firing_rates.T)

            if self.plastic_feedback_weights:
                top_down_inputs = np.dot(feedback_weights, next_pyramidal_firing_rates)
                change_feedback_weights = feedback_learning_rate *\
                                          np.dot((self.transfer_function(pyramidal_somatic_potentials) -
                                                  self.transfer_function(top_down_inputs)),
                                                 next_pyramidal_firing_rates.T)
            else:
                change_feedback_weights = np.zeros(feedforward_weights.shape)

            self.change_pyramidal_somatic_potentials += change_pyramidal_somatic_potentials
            self.change_interneuron_somatic_potentials += change_interneuron_somatic_potentials
            self.new_pyramidal_basal_potentials += new_pyramidal_basal_potentials
            self.new_pyramidal_apical_potentials += new_pyramidal_apical_potentials
            self.new_interneuron_basal_potential += new_interneuron_basal_potential

            self.change_feedforward_weights += change_feedforward_weights
            self.change_predict_weights += change_predict_weights
            self.change_interneuron_weights += change_interneuron_weights
            self.change_feedback_weights += change_feedback_weights'''








