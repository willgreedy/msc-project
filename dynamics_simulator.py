import numpy as np
from helpers import create_transfer_function

class DynamicsSimulator:
    def __init__(self, model, parameters):
        self.model = model
        self.iter_step = 0

        self.parameters = parameters
        self.leak_conductance = self.parameters['leak_conductance']
        self.apical_conductance = self.parameters['apical_conductance']
        self.basal_conductance = self.parameters['basal_conductance']
        self.dendritic_conductance = self.parameters['dendritic_conductance']
        self.nudging_conductance = self.parameters['nudging_conductance']
        self.background_noise_std = self.parameters['background_noise_std']
        self.transfer_function = create_transfer_function(self.parameters['transfer_function'])

    def step_simulation(self):
        self.iter_step += 1

        # input = self.data_stream.get_input(self.iter_step)
        input = np.zeros((self.model.input_size,))

        # output = self.data_stream.get_output(self.iter_step)
        output = np.zeros((self.model.output_size,))

        layers = self.model.get_layers()

        # Update hidden layer potentials
        for layer in layers[:-1]:
            pyramidal_somatic_potentials = layer.get_pyramidal_somatic_potentials()
            pyramidal_basal_potentials = layer.get_pyramidal_basal_potentials()
            pyramidal_apical_potentials = layer.get_pyramidal_apical_potentials()

            interneuron_somatic_potentials = layer.get_interneuron_somatic_potentials()
            interneuron_basal_potentials = layer.get_interneuron_basal_potentials()

            feedback_weights = layer.get_feedback_weights()
            feedforward_weights = layer.get_feedforward_weights()

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

            new_pyramidal_basal_potentials = np.dot(feedback_weights, self.transfer_function()