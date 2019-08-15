import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, load_append_data, get_smoothed_data

noiseless_layer_1_weight_magnitudes_iters, noiseless_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noiseless/layer_1_feedforward_predict_weight_angle', 3)
noise_0_15_layer_1_weight_magnitudes_iters, noise_0_15_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noise0-15/layer_1_feedforward_predict_weight_angle', 3)
noise_0_30_layer_1_weight_magnitudes_iters, noise_0_30_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noise0-30/layer_1_feedforward_predict_weight_angle', 3)
noise_0_45_layer_1_weight_magnitudes_iters, noise_0_45_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noise0-45/layer_1_feedforward_predict_weight_angle', 3)
noise_0_60_layer_1_weight_magnitudes_iters, noise_0_60_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noise0-60/layer_1_feedforward_predict_weight_angle', 3)
noise_0_90_layer_1_weight_magnitudes_iters, noise_0_90_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noise0-90/layer_1_feedforward_predict_weight_angle', 3)
noise_1_20_layer_1_weight_magnitudes_iters, noise_1_20_layer_1_weight_magnitudes_vals = load_append_data('./raw_data/noise1-20/layer_1_feedforward_predict_weight_angle', 3)

num_vals = len(noiseless_layer_1_weight_magnitudes_iters) - 15000

smoothed_noiseless_layer_1_weight_magnitudes_values = get_smoothed_data(noiseless_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_15_layer_1_weight_magnitudes_values = get_smoothed_data(noise_0_15_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_30_layer_1_weight_magnitudes_values = get_smoothed_data(noise_0_30_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_45_layer_1_weight_magnitudes_values = get_smoothed_data(noise_0_45_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_60_layer_1_weight_magnitudes_values = get_smoothed_data(noise_0_60_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_90_layer_1_weight_magnitudes_values = get_smoothed_data(noise_0_90_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_1_20_layer_1_weight_magnitudes_values = get_smoothed_data(noise_1_20_layer_1_weight_magnitudes_vals[:num_vals:5], 35000)
print("1")

plt.figure()
plt.plot(noiseless_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noiseless_layer_1_weight_magnitudes_values, label='No noise')
print('p')
plt.plot(noise_0_15_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noise_0_15_layer_1_weight_magnitudes_values, label='$\sigma = 0.15$')
print('p')
plt.plot(noise_0_30_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noise_0_30_layer_1_weight_magnitudes_values, label='$\sigma = 0.30$')
print('p')
plt.plot(noise_0_45_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noise_0_45_layer_1_weight_magnitudes_values, label='$\sigma = 0.45$')
print('p')
plt.plot(noise_0_60_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noise_0_60_layer_1_weight_magnitudes_values, label='$\sigma = 0.60$')
print('p')
plt.plot(noise_0_90_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noise_0_90_layer_1_weight_magnitudes_values, label='$\sigma = 0.90$')
print('p')
plt.plot(noise_1_20_layer_1_weight_magnitudes_iters[:num_vals:5], smoothed_noise_1_20_layer_1_weight_magnitudes_values, label='$\sigma = 1.20$')
print('p')

plt.legend()
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\mathbf{W}_{1, 2}^{\mathrm{PP}}, \mathbf{W}_{1, 1}^{\mathrm{IP}})$ (degrees)', size=12)
plt.savefig('figure6-predict.pdf', bbox_inches='tight')

#plt.show()


