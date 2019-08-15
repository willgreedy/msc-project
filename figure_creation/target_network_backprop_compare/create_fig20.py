import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_smoothed_data

layer_1_backprop_angle_iters, layer_1_backprop_angle_vals = load_object('./raw_data/layer_1_backprop_update_angle.pkl')
smoothed_layer_1_backprop_angle_vals = get_smoothed_data(layer_1_backprop_angle_vals, 200)

layer_2_backprop_angle_iters, layer_2_backprop_angle_vals = load_object('./raw_data/layer_2_backprop_update_angle.pkl')
smoothed_layer_2_backprop_angle_vals = get_smoothed_data(layer_2_backprop_angle_vals, 200)

plt.figure()
plt.xlim((0, layer_1_backprop_angle_iters[-1]))
plt.ylim((40, 140))
plt.plot(layer_1_backprop_angle_iters, layer_1_backprop_angle_vals, c='b')
plt.plot(layer_1_backprop_angle_iters, smoothed_layer_1_backprop_angle_vals, c='r')
plt.plot(layer_1_backprop_angle_iters, [90 for a in layer_2_backprop_angle_iters], c='#bf5c00', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\delta_{BP}, \delta_{DEN})$', size=12)
plt.savefig('figure20-l.pdf', bbox_inches='tight')

plt.figure()
plt.xlim((0, layer_2_backprop_angle_iters[-1]))
plt.ylim((0, 100))
plt.plot(layer_2_backprop_angle_iters, layer_2_backprop_angle_vals, c='b')
plt.plot(layer_2_backprop_angle_iters, smoothed_layer_2_backprop_angle_vals, c='r')
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\delta_{BP}, \delta_{DEN})$', size=12)
plt.savefig('figure20-r.pdf', bbox_inches='tight')

plt.show()