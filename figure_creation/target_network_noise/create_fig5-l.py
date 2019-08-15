import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_append_data, get_smoothed_data

noiseless_error_raw_vals_iters, noiseless_error_raw_vals_vals = load_append_data('./raw_data/noiseless/sum_squares_error', 3)
noise_0_15_error_raw_vals_iters, noise_0_15_error_raw_vals_vals = load_append_data('./raw_data/noise0-15/sum_squares_error', 3)
noise_0_30_error_raw_vals_iters, noise_0_30_error_raw_vals_vals = load_append_data('./raw_data/noise0-30/sum_squares_error', 3)
noise_0_45_error_raw_vals_iters, noise_0_45_error_raw_vals_vals = load_append_data('./raw_data/noise0-45/sum_squares_error', 3)
noise_0_60_error_raw_vals_iters, noise_0_60_error_raw_vals_vals = load_append_data('./raw_data/noise0-60/sum_squares_error', 3)
noise_0_90_error_raw_vals_iters, noise_0_90_error_raw_vals_vals = load_append_data('./raw_data/noise0-90/sum_squares_error', 3)
noise_1_20_error_raw_vals_iters, noise_1_20_error_raw_vals_vals = load_append_data('./raw_data/noise1-20/sum_squares_error', 3)

num_vals = len(noiseless_error_raw_vals_iters) - 15000

smoothed_noiseless_train_error_values = get_smoothed_data(noiseless_error_raw_vals_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_15_train_error_values = get_smoothed_data(noise_0_15_error_raw_vals_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_30_train_error_values = get_smoothed_data(noise_0_30_error_raw_vals_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_45_train_error_values = get_smoothed_data(noise_0_45_error_raw_vals_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_60_train_error_values = get_smoothed_data(noise_0_60_error_raw_vals_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_0_90_train_error_values = get_smoothed_data(noise_0_90_error_raw_vals_vals[:num_vals:5], 35000)
print("1")
smoothed_noise_1_20_train_error_values = get_smoothed_data(noise_1_20_error_raw_vals_vals[:num_vals:5], 35000)
print("1")

plt.figure()
plt.plot(noiseless_error_raw_vals_iters[:num_vals:5], smoothed_noiseless_train_error_values, label='No noise')
print('p')
plt.plot(noise_0_15_error_raw_vals_iters[:num_vals:5], smoothed_noise_0_15_train_error_values, label='$\sigma = 0.15$')
print('p')
plt.plot(noise_0_30_error_raw_vals_iters[:num_vals:5], smoothed_noise_0_30_train_error_values, label='$\sigma = 0.30$')
print('p')
plt.plot(noise_0_45_error_raw_vals_iters[:num_vals:5], smoothed_noise_0_45_train_error_values, label='$\sigma = 0.45$')
print('p')
plt.plot(noise_0_60_error_raw_vals_iters[:num_vals:5], smoothed_noise_0_60_train_error_values, label='$\sigma = 0.60$')
print('p')
plt.plot(noise_0_90_error_raw_vals_iters[:num_vals:5], smoothed_noise_0_90_train_error_values, label='$\sigma = 0.90$')
print('p')
plt.plot(noise_1_20_error_raw_vals_iters[:num_vals:5], smoothed_noise_1_20_train_error_values, label='$\sigma = 1.20$')
print('p')

plt.legend()
plt.ylim((0, 0.01 + max(smoothed_noiseless_train_error_values)))
plt.xlabel('Iterations')
plt.ylabel('Training Error')
plt.savefig('figure5-l.pdf', bbox_inches='tight')

#plt.show()


