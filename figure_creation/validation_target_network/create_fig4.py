import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_plot_data, get_smoothed_data

feedforward_iters, feedforward_vals = load_object('./raw_data/layer_1_feedforward_feedback_weight_angle.pkl')

print(np.min(feedforward_vals))

num_vals = len(feedforward_iters)
plt.figure()
plt.ylim((0, 100))
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.plot(feedforward_iters[:num_vals], feedforward_vals[:num_vals], c='b')
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\mathbf{W}_{1, 2}^{\mathrm{PP}}, \mathbf{W}_{2, 1}^{\mathrm{PP}})$ (degrees)', size=12)
plt.savefig('figure4.pdf', bbox_inches='tight')

plt.show()


