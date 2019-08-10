from layers import StandardLayer, OutputPyramidalLayer
import copy


class MultiCompartmentModel:
    def __init__(self, input_size, hidden_layer_sizes, output_size, feedforward_weight_intitialiser,
                 feedback_weight_intitialiser, predict_weight_intitialiser, interneuron_weight_intitialiser,
                 feedforward_learning_rates, predict_learning_rates, interneuron_learning_rates, feedback_learning_rates,
                 init_self_predicting_weights=False, self_predicting_scale_factor=None, tied_weights=False):

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        num_layers = len(layer_sizes)

        if len(feedforward_learning_rates) != num_layers - 1:
            raise Exception("Received incorrect number of feedforward learning rates. Received: {}, Expected: {}"
                            .format(len(feedforward_learning_rates), num_layers))
        if len(predict_learning_rates) != num_layers - 2:
            raise Exception("Received incorrect number of predict learning rates. Received: {}, Expected: {}"
                            .format(len(predict_learning_rates), num_layers - 1))
        if len(interneuron_learning_rates) != num_layers - 2:
            raise Exception("Received incorrect number of interneuron learning rates. Received: {}, Expected: {}"
                            .format(len(interneuron_learning_rates), num_layers - 1))
        if len(feedback_learning_rates) != num_layers - 2:
            raise Exception("Received incorrect number of feedback learning rates. Received: {}, Expected: {}"
                            .format(len(feedback_learning_rates), num_layers - 1))

        feedforward_weights = [feedforward_weight_intitialiser.sample((layer_sizes[i+1], layer_sizes[i])) for i in
                               range(len(layer_sizes) - 1)]

        if init_self_predicting_weights:
            if tied_weights:
                predict_weights = [ff_weights for ff_weights in feedforward_weights[1:]]
                interneuron_weights = [-ff_weights.T for ff_weights in feedforward_weights[1:]]
                feedback_weights = [ff_weights.T for ff_weights in feedforward_weights[1:]]
            else:
                predict_weights = [self_predicting_scale_factor * ff_weights.copy() for ff_weights in feedforward_weights[1:-1]] + [feedforward_weights[-1].copy()]
                interneuron_weights = [-ff_weights.copy().T for ff_weights in feedforward_weights[1:]]
                feedback_weights = [ff_weights.copy().T for ff_weights in feedforward_weights[1:]]
        else:
            predict_weights = [predict_weight_intitialiser.sample((layer_sizes[i+1], layer_sizes[i])) for i in
                               range(1, len(layer_sizes) - 1)]
            interneuron_weights = [interneuron_weight_intitialiser.sample((layer_sizes[i], layer_sizes[i+1])) for i in
                           range(1, len(layer_sizes) - 1)]
            feedback_weights = [feedback_weight_intitialiser.sample((layer_sizes[i], layer_sizes[i+1])) for i in
                               range(1, len(layer_sizes) - 1)]

        layers = []
        for i in range(0, len(layer_sizes) - 2):
            print(i)
            layers += [('StandardLayer_{}'.format(i+1), StandardLayer(layer_sizes[i], layer_sizes[i+1], layer_sizes[i+2],
                                                                      feedforward_weights[i],
                                                                      feedback_weights[i],
                                                                      predict_weights[i],
                                                                      interneuron_weights[i],
                                                                      feedforward_learning_rates[i],
                                                                      predict_learning_rates[i],
                                                                      interneuron_learning_rates[i],
                                                                      feedback_learning_rates[i]))]

        layers += [('OutputLayer', OutputPyramidalLayer(layer_sizes[-2], layer_sizes[-1],
                                                        feedforward_weights[-1],
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
