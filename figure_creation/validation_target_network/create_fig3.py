import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_plot_data, get_smoothed_data

predict_iters, predict_vals = load_object('./raw_data/layer_1_feedforward_predict_weight_angle.pkl')
iterneuron_iters, interneuron_vals = load_object('./raw_data/layer_1_feedback_interneuron_weight_angle.pkl')

print(predict_vals[-10:])
print(interneuron_vals[-10:])

num_vals = len(predict_iters)
plt.figure()
plt.ylim((0, 100))
plt.plot(predict_iters[:num_vals], predict_vals[:num_vals], c='b')
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\mathbf{W}_{1, 2}^{\mathrm{PP}}, \mathbf{W}_{1, 1}^{\mathrm{IP}})$ (degrees)', size=12)
plt.savefig('figure3-l.pdf', bbox_inches='tight')

num_vals = len(iterneuron_iters)
plt.figure()
plt.ylim((0, 180))
plt.plot(iterneuron_iters[:num_vals], interneuron_vals[:num_vals], c='b')
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\mathbf{W}_{2, 1}^{\mathrm{PP}}, \mathbf{W}_{1, 1}^{\mathrm{PI}})$ (degrees)', size=12)
plt.savefig('figure3-r.pdf', bbox_inches='tight')


plt.show()


