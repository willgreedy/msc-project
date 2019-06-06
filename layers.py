import numpy as np


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons


class StandardLayer(Layer):
    """Standard layer used for both input and hidden layer activity. Contains pyramidal and interneuron sub-layers.
    """
    def __init__(self, num_inputs, num_neurons, num_neurons_next, feedforward_weight_intitialiser,
                 feedback_weight_intitialiser, predict_weight_intitialiser, interneuron_weight_intitialiser):
        super(StandardLayer, self).__init__(num_neurons)

        self.pyramidal_sub_layer = PyramidalSubLayer(self.num_neurons)
        self.interneuron_sub_layer = InterneuronSubLayer(num_neurons_next)

        self.feedforward_weights = feedforward_weight_intitialiser.sample((self.num_neurons, num_inputs))
        self.feedback_weights = feedback_weight_intitialiser.sample((self.num_neurons, num_neurons_next))

        self.predict_weights = predict_weight_intitialiser((num_neurons_next, self.num_neurons))
        self.interneuron_weights = interneuron_weight_intitialiser((self.num_neurons, num_neurons_next))


class OutputPyramidalLayer(Layer):
    """Output layer includes no SST interneurons and has no apical compartments in pyramidal neurons.
    """
    def __init__(self, num_inputs, num_neurons, feedforward_weight_intitialiser):
        super(OutputPyramidalLayer, self).__init__(num_neurons)
        self.basal_potentials = np.zeros((self.num_neurons, ))
        self.soma_potentials = np.zeros((self.num_neurons,))

        self.feedforward_weights = feedforward_weight_intitialiser.sample((self.num_neurons, num_inputs))


class PyramidalSubLayer(Layer):
    def __init__(self, num_neurons):
        super(PyramidalSubLayer, self).__init__(num_neurons)
        self.apical_potentials = np.zeros((self.num_neurons,))
        self.basal_potentials = np.zeros((self.num_neurons, ))
        self.soma_potentials = np.zeros((self.num_neurons,))


class InterneuronSubLayer(Layer):
    def __init__(self, num_neurons):
        super(InterneuronSubLayer, self).__init__(num_neurons)
        self.basal_potentials = np.zeros((self.num_neurons,))
        self.soma_potentials = np.zeros((self.num_neurons,))