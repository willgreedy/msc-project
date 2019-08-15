import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_plot_data, get_smoothed_data

error_raw_iters, error_raw_vals = load_object('./raw_data/sum_squares_error.pkl')


num_vals = len(error_raw_iters) - 100000

smoothed_train_error_values = get_smoothed_data(error_raw_vals[:num_vals], 35000)
plt.figure()
plt.ylim((0, max(smoothed_train_error_values) + 0.01))
plt.plot(error_raw_iters[:num_vals], smoothed_train_error_values, c='b')
plt.xlabel('Iterations')
plt.ylabel('Training Error')
plt.savefig('figure2.pdf', bbox_inches='tight')

plt.show()


