import numpy as np
from abc import ABC


class Layer(ABC):
    def __init__(self, num_inputs, num_neurons, feedforward_weight_intitialiser, feedforward_learning_rate):
        self.num_neurons = num_neurons

        self.pyramidal_somatic_potentials = np.zeros((self.num_neurons, 1))
        self.pyramidal_basal_potentials = np.zeros((self.num_neurons, 1))

        self.feedforward_weights = feedforward_weight_intitialiser.sample((self.num_neurons, num_inputs))
        self.feedforward_learning_rate = feedforward_learning_rate

        self.change_pyramidal_somatic_potentials = None
        self.new_pyramidal_basal_potentials = None
        self.change_feedforward_weights = None

    def perform_update(self, step_size):
        self.pyramidal_somatic_potentials += step_size * self.change_pyramidal_somatic_potentials
        self.pyramidal_basal_potentials = self.new_pyramidal_basal_potentials
        self.feedforward_weights += step_size * self.change_feedforward_weights

    def reset_stored_updates(self):
        self.change_pyramidal_somatic_potentials = None
        self.new_pyramidal_basal_potentials = None
        self.change_feedforward_weights = None

    def get_pyramidal_somatic_potentials(self):
        return self.pyramidal_somatic_potentials

    def get_pyramidal_basal_potentials(self):
        return self.pyramidal_basal_potentials

    def get_feedforward_weights(self):
        return self.feedforward_weights

    def get_feedforward_learning_rate(self):
        return self.feedforward_learning_rate

    def set_change_pyramidal_somatic_potentials(self, change_pyramidal_somatic_potentials):
        self.change_pyramidal_somatic_potentials = change_pyramidal_somatic_potentials

    def set_new_pyramidal_basal_potentials(self, new_pyramidal_basal_potentials):
        self.new_pyramidal_basal_potentials = new_pyramidal_basal_potentials

    def set_change_feedforward_weights(self, change_feedforward_weights):
        self.change_feedforward_weights = change_feedforward_weights


class StandardLayer(Layer):
    """Standard layer used for both input and hidden layer activity. Contains pyramidal and interneuron sub-layers.
    """
    def __init__(self, num_inputs, num_neurons, num_neurons_next, feedforward_weight_intitialiser,
                 feedback_weight_intitialiser, predict_weight_intitialiser, interneuron_weight_intitialiser,
                 feedforward_learning_rate, predict_learning_rate, interneuron_learning_rate, feedback_learning_rate):
        super(StandardLayer, self).__init__(num_inputs, num_neurons, feedforward_weight_intitialiser,
                                            feedforward_learning_rate)
        self.num_neurons_next = num_neurons_next

        self.pyramidal_apical_potentials = np.zeros((self.num_neurons, 1))

        self.interneuron_basal_potentials = np.zeros((self.num_neurons_next, 1))
        self.interneuron_somatic_potentials = np.zeros((self.num_neurons_next, 1))

        self.feedback_weights = feedback_weight_intitialiser.sample((self.num_neurons, self.num_neurons_next))
        self.predict_weights = predict_weight_intitialiser.sample((self.num_neurons_next, self.num_neurons))
        self.interneuron_weights = interneuron_weight_intitialiser.sample((self.num_neurons, self.num_neurons_next))

        self.predict_learning_rate = predict_learning_rate
        self.interneuron_learning_rate = interneuron_learning_rate
        self.feedback_learning_rate = feedback_learning_rate

        self.change_interneuron_somatic_potentials = None
        self.new_pyramidal_apical_potentials = None
        self.new_interneuron_basal_potentials = None

        self.change_predict_weights = None
        self.change_interneuron_weights = None
        self.change_feedback_weights = None

    def perform_update(self, step_size):
        super().perform_update(step_size)

        self.interneuron_somatic_potentials += step_size * self.change_interneuron_somatic_potentials
        self.pyramidal_apical_potentials = self.new_pyramidal_apical_potentials
        self.interneuron_basal_potentials = self.new_interneuron_basal_potentials

        self.feedback_weights += step_size * self.change_feedback_weights
        self.predict_weights += step_size * self.change_predict_weights
        self.interneuron_weights += step_size * self.change_interneuron_weights

    def reset_stored_updates(self):
        self.change_interneuron_somatic_potentials = None
        self.new_pyramidal_apical_potentials = None
        self.new_interneuron_basal_potentials = None

        self.change_predict_weights = None
        self.change_interneuron_weights = None
        self.change_feedback_weights = None

    def get_pyramidal_apical_potentials(self):
        return self.pyramidal_apical_potentials

    def get_interneuron_basal_potentials(self):
        return self.interneuron_basal_potentials

    def get_interneuron_somatic_potentials(self):
        return self.interneuron_somatic_potentials

    def get_feedback_weights(self):
        return self.feedback_weights

    def get_predict_weights(self):
        return self.predict_weights

    def get_interneuron_weights(self):
        return self.interneuron_weights

    def get_predict_learning_rate(self):
        return self.predict_learning_rate

    def get_interneuron_learning_rate(self):
        return self.interneuron_learning_rate

    def get_feedback_learning_rate(self):
        return self.feedback_learning_rate

    def set_change_pyramidal_somatic_potentials(self, change_pyramidal_somatic_potentials):
        self.change_pyramidal_somatic_potentials = change_pyramidal_somatic_potentials

    def set_change_interneuron_somatic_potentials(self, change_interneuron_somatic_potentials):
        self.change_interneuron_somatic_potentials = change_interneuron_somatic_potentials

    def set_new_pyramidal_basal_potentials(self, new_pyramidal_basal_potentials):
        self.new_pyramidal_basal_potentials = new_pyramidal_basal_potentials

    def set_new_pyramidal_apical_potentials(self, new_pyramidal_apical_potentials):
        self.new_pyramidal_apical_potentials = new_pyramidal_apical_potentials

    def set_new_interneuron_basal_potentials(self, new_interneuron_basal_potentials):
        self.new_interneuron_basal_potentials = new_interneuron_basal_potentials

    def set_change_feedforward_weights(self, change_feedforward_weights):
        self.change_feedforward_weights = change_feedforward_weights

    def set_change_predict_weights(self, change_predict_weights):
        self.change_predict_weights = change_predict_weights

    def set_change_interneuron_weights(self, change_interneuron_weights):
        self.change_interneuron_weights = change_interneuron_weights

    def set_change_feedback_weights(self, change_feedback_weights):
        self.change_feedback_weights = change_feedback_weights


class OutputPyramidalLayer(Layer):
    """Output layer includes no SST interneurons and has no apical compartments in pyramidal neurons.
    """
    def __init__(self, num_inputs, num_neurons, feedforward_weight_intitialiser, feedforward_learning_rate):
        super(OutputPyramidalLayer, self).__init__(num_inputs, num_neurons, feedforward_weight_intitialiser,
                                                   feedforward_learning_rate)
