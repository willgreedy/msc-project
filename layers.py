class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons


class StandardLayer(Layer):
    """Standard layer used for both input and hidden layer activity.
    """
    def __init__(self, num_neurons):
        super(StandardLayer, self).__init__(num_neurons)


class OutputLayer(Layer):
    """Output layer includes no SST interneurons and has no apical compartments in pyramidal neurons.
    """
    def __init__(self, num_neurons):
        super(OutputLayer, self).__init__(num_neurons)