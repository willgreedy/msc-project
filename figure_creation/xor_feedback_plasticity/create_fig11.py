import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object

feedforward_feedback_fb_plas_0_001_vals_iters, feedforward_feedback_fb_plas_0_001_vals = load_object('./raw_data/fb_plas_0-001/layer_1_feedforward_feedback_weight_angle.pkl')
#feedforward_feedback_fb_plas_0_00075_vals_iters, feedforward_feedback_fb_plas_0_00075_vals = load_object('./raw_data/fb_plas_0-00075/layer_1_feedforward_feedback_weight_angle.pkl')
feedforward_feedback_fb_plas_0_0005_vals_iters, feedforward_feedback_fb_plas_0_0005_vals = load_object('./raw_data/fb_plas_0-0005/layer_1_feedforward_feedback_weight_angle.pkl')
feedforward_feedback_fb_plas_0_00025_vals_iters, feedforward_feedback_fb_plas_0_00025_vals = load_object('./raw_data/fb_plas_0-00025/layer_1_feedforward_feedback_weight_angle.pkl')
feedforward_feedback_fb_plas_0_000125_vals_iters, feedforward_feedback_fb_plas_0_000125_vals = load_object('./raw_data/fb_plas_0-000125/layer_1_feedforward_feedback_weight_angle.pkl')
feedforward_feedback_fb_plas_0_0000625_vals_iters, feedforward_feedback_fb_plas_0_0000625_vals = load_object('./raw_data/fb_plas_0-0000625/layer_1_feedforward_feedback_weight_angle.pkl')
feedforward_feedback_fb_align_vals_iters, feedforward_fb_align_feedback_vals = load_object('./raw_data/fb_align/layer_1_feedforward_feedback_weight_angle.pkl')

num_vals = len(feedforward_feedback_fb_align_vals_iters)

plt.figure()
    plt.plot(feedforward_feedback_fb_plas_0_001_vals_iters[:num_vals], feedforward_feedback_fb_plas_0_001_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 1.0 \mathrm{x} 10^{-3}$')
#plt.plot(feedforward_feedback_fb_plas_0_00075_vals_iters[:num_vals], feedforward_feedback_fb_plas_0_00075_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 0.00075$')
plt.plot(feedforward_feedback_fb_plas_0_0005_vals_iters[:num_vals], feedforward_feedback_fb_plas_0_0005_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 5.0 \mathrm{x} 10^{-4}$')
plt.plot(feedforward_feedback_fb_plas_0_00025_vals_iters[:num_vals], feedforward_feedback_fb_plas_0_00025_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 2.5 \mathrm{x} 10^{-4}$')
plt.plot(feedforward_feedback_fb_plas_0_000125_vals_iters[:num_vals], feedforward_feedback_fb_plas_0_000125_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 1.25 \mathrm{x} 10^{-4}$')
plt.plot(feedforward_feedback_fb_plas_0_0000625_vals_iters[:num_vals], feedforward_feedback_fb_plas_0_0000625_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 6.25 \mathrm{x} 10^{-5}$')
plt.plot(feedforward_feedback_fb_align_vals_iters[:num_vals], feedforward_fb_align_feedback_vals[:num_vals], label='Feedback Alignment')


plt.legend(loc='upper right', prop={'size': 9})
plt.ylim((0, 100))
plt.xlabel('Iterations')
plt.ylabel('$\\angle (\mathbf{W}_{1, 2}^{\mathrm{PP}}, \mathbf{W}_{2, 1}^{\mathrm{PP}})$')
plt.savefig('figure11.pdf', bbox_inches='tight')

plt.show()


