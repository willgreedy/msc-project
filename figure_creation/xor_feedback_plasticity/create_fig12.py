import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_smoothed_data

error_fb_plas_0_001_vals_iters, error_fb_plas_0_001_vals = load_object('./raw_data/fb_plas_0-001/sum_squares_error.pkl')
#error_fb_plas_0_00075_vals_iters, error_fb_plas_0_00075_vals = load_object('./raw_data/fb_plas_0-00075/sum_squares_error.pkl')
error_fb_plas_0_0005_vals_iters, error_fb_plas_0_0005_vals = load_object('./raw_data/fb_plas_0-0005/sum_squares_error.pkl')
error_fb_plas_0_00025_vals_iters, error_fb_plas_0_00025_vals = load_object('./raw_data/fb_plas_0-00025/sum_squares_error.pkl')
error_fb_plas_0_000125_vals_iters, error_fb_plas_0_000125_vals = load_object('./raw_data/fb_plas_0-000125/sum_squares_error.pkl')
error_fb_plas_0_0000625_vals_iters, error_fb_plas_0_0000625_vals = load_object('./raw_data/fb_plas_0-0000625/sum_squares_error.pkl')
error_fb_plas_0_00003125_vals_iters, error_fb_plas_0_00003125_vals = load_object('./raw_data/fb_plas_0-00003125/sum_squares_error.pkl')
error_fb_align_vals_iters, error_fb_align_feedback_vals = load_object('./raw_data/fb_align/sum_squares_error.pkl')

error_fb_plas_0_001_vals = get_smoothed_data(error_fb_plas_0_001_vals, 2500)
error_fb_plas_0_0005_vals = get_smoothed_data(error_fb_plas_0_0005_vals, 2500)
error_fb_plas_0_00025_vals = get_smoothed_data(error_fb_plas_0_00025_vals, 2500)
error_fb_plas_0_000125_vals = get_smoothed_data(error_fb_plas_0_000125_vals, 2500)
error_fb_plas_0_0000625_vals = get_smoothed_data(error_fb_plas_0_0000625_vals, 2500)
error_fb_plas_0_00003125_vals = get_smoothed_data(error_fb_plas_0_00003125_vals, 2500)
error_fb_align_feedback_vals = get_smoothed_data(error_fb_align_feedback_vals, 2500)

num_vals = len(error_fb_align_vals_iters)

plt.figure()
plt.plot(error_fb_plas_0_001_vals_iters[:num_vals], error_fb_plas_0_001_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 1.0 \mathrm{x} 10^{-3}$')
#plt.plot(error_fb_plas_0_00075_vals_iters[:num_vals], error_fb_plas_0_00075_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 0.00075$')
plt.plot(error_fb_plas_0_0005_vals_iters[:num_vals], error_fb_plas_0_0005_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 5.0 \mathrm{x} 10^{-4}$')
plt.plot(error_fb_plas_0_00025_vals_iters[:num_vals], error_fb_plas_0_00025_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 2.5 \mathrm{x} 10^{-4}$')
plt.plot(error_fb_plas_0_000125_vals_iters[:num_vals], error_fb_plas_0_000125_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 1.25 \mathrm{x} 10^{-4}$')
plt.plot(error_fb_plas_0_0000625_vals_iters[:num_vals], error_fb_plas_0_0000625_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 6.25 \mathrm{x} 10^{-5}$')
#plt.plot(error_fb_plas_0_00003125_vals_iters[:num_vals], error_fb_plas_0_00003125_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 3.125 \mathrm{x} 10^{-5}$')
plt.plot(error_fb_align_vals_iters[:num_vals], error_fb_align_feedback_vals[:num_vals], label='Feedback Alignment')


plt.legend(loc='lower left', prop={'size': 9})
plt.xlabel('Iterations')
plt.ylabel('Training Error')
plt.ylim((0, max(error_fb_align_feedback_vals)+0.002))
plt.savefig('figure12.pdf', bbox_inches='tight')

plt.show()


