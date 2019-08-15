import matplotlib.pyplot as plt
import numpy as np
from figure_creation.figure_helpers import load_object, get_smoothed_data

fb_plas_0_001_model = load_object('./raw_data/fb_plas_0-001/xor_sigmoid_fb_plas_0-001.pkl')
fb_plas_0_0005_model = load_object('./raw_data/fb_plas_0-0005/xor_sigmoid_fb_plas_0-0005.pkl')
fb_plas_0_00025_model = load_object('./raw_data/fb_plas_0-00025/xor_sigmoid_fb_plas_0-00025.pkl')
fb_plas_0_000125_model = load_object('./raw_data/fb_plas_0-000125/xor_sigmoid_fb_plas_0-000125.pkl')
fb_plas_0_0000625_model = load_object('./raw_data/fb_plas_0-0000625/xor_sigmoid_fb_plas_0-0000625.pkl')
fb_plas_0_00003125_model = load_object('./raw_data/fb_plas_0-00003125/xor_sigmoid_fb_plas_0-00003125.pkl')
fb_align_model = load_object('./raw_data/fb_align/xor_sigmoid_fb_align.pkl')

models = [fb_plas_0_001_model, fb_plas_0_0005_model, fb_plas_0_00025_model, fb_plas_0_000125_model, fb_plas_0_0000625_model, fb_plas_0_00003125_model, fb_align_model]
model_ff_weights = [model.get_layers()[1][1].get_feedforward_weights() for model in models]
model_fb_weights = [model.get_layers()[0][1].get_feedback_weights() for model in models]


for model_ff_weight, model_fb_weight in zip(model_ff_weights, model_fb_weights):
    #plt.figure()
    #print(len(model_fb_weight))
    #print("Mean: {}, Median: {}, Std: {}, Max: {}, Min: {}".format(np.mean(model_ff_weight), np.median(model_ff_weight),
    #                                                               np.std(model_ff_weight), np.max(model_ff_weight),
    #                                                               np.min(model_ff_weight)))
    #print("Mean: {}, Median: {}, Std: {}, Max: {}, Min: {}".format(np.mean(model_fb_weight), np.median(model_fb_weight),
    #                                                               np.std(model_fb_weight), np.max(model_fb_weight),
    #                                                               np.min(model_fb_weight)))

    max_loc = np.argmax(model_fb_weight)
    print(max_loc)
    print(model_fb_weight.T)
    print(model_ff_weight)
    print()
    #a = model_fb_weight.flatten()
    #a = a[a < 1.0]
    #plt.hist(a)#, bins=[0.1*i-1.0 for i in range(20)])

plt.show()

'''
plt.figure()
plt.plot(error_fb_plas_0_001_vals_iters[:num_vals], error_fb_plas_0_001_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 1.0 \mathrm{x} 10^{-3}$')
plt.plot(error_fb_plas_0_0005_vals_iters[:num_vals], error_fb_plas_0_0005_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 5.0 \mathrm{x} 10^{-4}$')
plt.plot(error_fb_plas_0_00025_vals_iters[:num_vals], error_fb_plas_0_00025_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 2.5 \mathrm{x} 10^{-4}$')
plt.plot(error_fb_plas_0_000125_vals_iters[:num_vals], error_fb_plas_0_000125_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 1.25 \mathrm{x} 10^{-4}$')
plt.plot(error_fb_plas_0_0000625_vals_iters[:num_vals], error_fb_plas_0_0000625_vals[:num_vals], label='$\eta_{1, 2}^{\mathrm{PP}} = 6.25 \mathrm{x} 10^{-5}$')
plt.plot(error_fb_align_vals_iters[:num_vals], error_fb_align_feedback_vals[:num_vals], label='Feedback Alignment')


plt.legend(loc='lower left', prop={'size': 9})
plt.xlabel('Iterations')
plt.ylabel('Training Error')
plt.ylim((0, max(error_fb_align_feedback_vals)+0.002))
plt.savefig('figure12.pdf', bbox_inches='tight')

plt.show()

'''
