from layers import StandardLayer, OutputPyramidalLayer


class MultiCompartmentModel:
    def __init__(self, input_size, layer_sizes, output_size, feedforward_weight_intitialiser,
                 feedback_weight_intitialiser, predict_weight_intitialiser, interneuron_weight_intitialiser,
                 feedforward_learning_rate, predict_learning_rate, interneuron_learning_rate, feedback_learning_rate):
        layers = []

        layers += [('StandardLayer_1', StandardLayer(input_size, layer_sizes[0], layer_sizes[1],
                                                     feedforward_weight_intitialiser,
                                                     feedback_weight_intitialiser,
                                                     predict_weight_intitialiser,
                                                     interneuron_weight_intitialiser,
                                                     feedforward_learning_rate,
                                                     predict_learning_rate,
                                                     interneuron_learning_rate,
                                                     feedback_learning_rate))]

        for i in range(1, len(layer_sizes) - 1):
            layers += [('StandardLayer_{}'.format(i), StandardLayer(layer_sizes[i-1], layer_sizes[i], layer_sizes[i+1],
                                                                    feedforward_weight_intitialiser,
                                                                    feedback_weight_intitialiser,
                                                                    predict_weight_intitialiser,
                                                                    interneuron_weight_intitialiser,
                                                                    feedforward_learning_rate,
                                                                    predict_learning_rate,
                                                                    interneuron_learning_rate,
                                                                    feedback_learning_rate))]

        layers += [('OutputLayer', OutputPyramidalLayer(layer_sizes[-2], output_size,
                                                        feedforward_weight_intitialiser,
                                                        feedforward_learning_rate))]

        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size

    def get_layers(self):
        return self.layers