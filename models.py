from layers import StandardLayer, OutputPyramidalLayer


class MultiCompartmentModel:
    def __init__(self, input_size, layer_sizes, output_size, feedforward_weight_intitialiser,
                 feedback_weight_intitialiser, predict_weight_intitialiser, interneuron_weight_intitialiser,
                 feedforward_learning_rates, predict_learning_rates, interneuron_learning_rates, feedback_learning_rates):

        num_layers = len(layer_sizes)
        if len(feedforward_learning_rates) != num_layers:
            raise Exception("Received incorrect number of feedforward learning rates. Received: {}, Expected: {}"
                            .format(len(feedforward_learning_rates), num_layers))
        if len(predict_learning_rates) != num_layers - 1:
            raise Exception("Received incorrect number of predict learning rates. Received: {}, Expected: {}"
                            .format(len(predict_learning_rates), num_layers - 1))
        if len(interneuron_learning_rates) != num_layers - 1:
            raise Exception("Received incorrect number of interneuron learning rates. Received: {}, Expected: {}"
                            .format(len(interneuron_learning_rates), num_layers - 1))
        if len(feedback_learning_rates) != num_layers - 1:
            raise Exception("Received incorrect number of feedback learning rates. Received: {}, Expected: {}"
                            .format(len(feedback_learning_rates), num_layers - 1))
        layers = []

        layers += [('StandardLayer_1', StandardLayer(input_size, layer_sizes[0], layer_sizes[1],
                                                     feedforward_weight_intitialiser,
                                                     feedback_weight_intitialiser,
                                                     predict_weight_intitialiser,
                                                     interneuron_weight_intitialiser,
                                                     feedforward_learning_rates[0],
                                                     predict_learning_rates[0],
                                                     interneuron_learning_rates[0],
                                                     feedback_learning_rates[0]))]

        for i in range(1, len(layer_sizes) - 1):
            layers += [('StandardLayer_{}'.format(i+1), StandardLayer(layer_sizes[i-1], layer_sizes[i], layer_sizes[i+1],
                                                                      feedforward_weight_intitialiser,
                                                                      feedback_weight_intitialiser,
                                                                      predict_weight_intitialiser,
                                                                      interneuron_weight_intitialiser,
                                                                      feedforward_learning_rates[i],
                                                                      predict_learning_rates[i],
                                                                      interneuron_learning_rates[i],
                                                                      feedback_learning_rates[i]))]

        layers += [('OutputLayer', OutputPyramidalLayer(layer_sizes[-2], output_size,
                                                        feedforward_weight_intitialiser,
                                                        feedforward_learning_rates[-1]))]

        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size

    def get_layers(self):
        return self.layers

    def __str__(self):
        model_string = "MultiCompartment model with {} layers.".format(len(self.layers))
        model_string += "\n"
        model_string += "Input size = " + str(self.input_size)
        for layer_name, layer in self.layers:
            model_string += "\n"
            model_string += layer_name
            model_string += ": "
            model_string += str(layer.num_neurons) + " neurons"
        model_string += "\n"
        model_string += "Output size = " + str(self.output_size)

        return model_string