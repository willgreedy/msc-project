from layers import StandardLayer, OutputPyramidalLayer


class MultiCompartmentModel:
    def __init__(self, input_size, layer_sizes, output_size, feedforward_weight_intitialiser,
                 feedback_weight_intitialiser, predict_weight_intitialiser, interneuron_weight_intitialiser):
        layers = []

        layers += StandardLayer(input_size, layer_sizes[0], layer_sizes[1], feedforward_weight_intitialiser,
                                feedback_weight_intitialiser, predict_weight_intitialiser,
                                interneuron_weight_intitialiser)

        for i in range(1, len(layer_sizes) - 1):
            layers += StandardLayer(layer_sizes[i-1], layer_sizes[i], layer_sizes[i+1],
                                    feedforward_weight_intitialiser, feedback_weight_intitialiser,
                                    predict_weight_intitialiser, interneuron_weight_intitialiser)

        layers += [OutputPyramidalLayer(layer_sizes[-1], output_size, feedforward_weight_intitialiser)]

        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size