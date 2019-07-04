import numpy as np

scale_factor1 = 2.0
scale_factor2 = 10.0

first_layer_feedforward_weights = np.random.uniform(-scale_factor1, scale_factor1, (30, 20))
second_layer_feedforward_weights = np.random.uniform(-scale_factor2, scale_factor2, (20, 10))

np.save("first_layer_feedforward_weights", first_layer_feedforward_weights)
np.save("second_layer_feedforward_weights", second_layer_feedforward_weights)