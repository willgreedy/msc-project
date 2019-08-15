import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_plot_data

target_iters, target_vals = load_object('./raw_data/individual_target_value.pkl')
soma_iters, soma_vals = load_object('./raw_data/layer_2_individual_pyramidal_soma_potential.pkl')
apical_iters, apical_vals = load_object('./raw_data/layer_1_individual_pyramidal_apical_potential.pkl')

plt.figure()
plt.plot(soma_iters, soma_vals, label='Output', c='b')
plt.plot(target_iters, target_vals, label='Target', c='r', linestyle='--')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Somatic Potential')

plt.savefig('figure1-l.pdf', bbox_inches='tight')

num_vals = 500
plt.figure()
plt.plot(apical_iters[:num_vals], apical_vals[:num_vals], label='Apical Potential', c='b')
plt.plot(apical_iters[:num_vals], np.zeros(num_vals,), c='black', linestyle='--')
#plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Apical Potential')
plt.savefig('figure1-r.pdf', bbox_inches='tight')

plt.show()