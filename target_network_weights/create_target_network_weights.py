import numpy as np
import pathlib

num_layers = 4

scale_factor1 = 5.0
scale_factor2 = 10.0
scale_factor3 = 15.0

input_size = 20
output_size = 10

hidden_layer1_size = 15
hidden_layer2_size = 15

first_layer_shape = (input_size, hidden_layer1_size)
second_layer_shape = (hidden_layer1_size, hidden_layer2_size)
third_layer_shape = (hidden_layer2_size, output_size)

first_layer_feedforward_weights = np.random.uniform(-scale_factor1, scale_factor1, first_layer_shape)
second_layer_feedforward_weights = np.random.uniform(-scale_factor2, scale_factor2, second_layer_shape)
third_layer_feedforward_weights = np.random.uniform(-scale_factor3, scale_factor3, third_layer_shape)


folder = './{}_layer_sf_{}x{}x{}'.format(num_layers, int(scale_factor1), int(scale_factor2), int(scale_factor3))
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

np.save("{}/layer1_weights".format(folder),
        first_layer_feedforward_weights)

np.save("{}/layer2_weights".format(folder),
        second_layer_feedforward_weights)

np.save("{}/layer3_weights".format(folder),
        third_layer_feedforward_weights)