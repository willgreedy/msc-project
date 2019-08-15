import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_plot_data, get_smoothed_data

target_iters, target_vals = load_object('./raw_data/individual_target_value.pkl')
output_iters, output_vals = load_object('./raw_data/layer_2_individual_pyramidal_basal_potential.pkl')

print(len(target_iters))
print(len(output_iters))

num_vals = len(target_iters)
min_index_start = 1000
max_index_start = 1500

min_index_test = num_vals - 500
max_index_test = num_vals

#fig = plt.figure()

fig, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2, figsize=(8, 6), sharex='col')

ax_3.set_xlabel('Iterations')
ax_4.set_xlabel('Iterations')

fig.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=1.0, wspace=0.05, hspace=0.00)

l1, = ax_1.plot(target_iters[min_index_start:max_index_start], target_vals[min_index_start:max_index_start], label='Target', c='r')
l2, = ax_2.plot(target_iters[min_index_test:max_index_test], target_vals[min_index_test:max_index_test], label='Target', c='r')
l3, = ax_3.plot(output_iters[min_index_start:max_index_start], output_vals[min_index_start:max_index_start], label='Output', c='b')
l4, = ax_4.plot(output_iters[min_index_test:max_index_test], output_vals[min_index_test:max_index_test], label='Output', c='b')

ax_2.legend([l1, l3], ["Target", "Output"], loc='upper right')

ax_1.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')
ax_3.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')
ax_2.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')
ax_4.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')

plt.savefig('figure10.pdf', bbox_inches='tight')

plt.show()


