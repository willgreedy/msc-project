import os
from shutil import copyfile

folder_base = 'D:/MSc Plots/Deltic Plots/experiment_results/target_network_lrx100_'
folders = ['noiseless', 'noise0-15', 'noise0-30', 'noise0-45', 'noise0-60', 'noise0-90']
#folders = ['noise1-20']

sub_folders = ['/epoch_500', '/epoch_750', '/epoch_1000', '/test_1250']
#sub_folders = ['/epoch_500', '/epoch_750', '/test_1000']

file_base = '/monitors/'
files = ['layer_1_std_feedforward_weight_magnitudes', 'layer_2_std_feedforward_weight_magnitudes']#['sum_squares_error', 'layer_1_mean_feedforward_weight_magnitudes', 'layer_1_mean_feedforward_weight_magnitudes', 'layer_2_mean_feedforward_weight_magnitudes', 'layer_2_individual_pyramidal_soma_potential', 'layer_1_feedback_interneuron_weight_angle', 'layer_1_feedforward_predict_weight_angle', 'layer_1_feedforward_feedback_weight_angle']

save_location_base = 'C:/Users/Will/PycharmProjects/MScProject/figure_creation/target_network_noise/raw_data/'

for folder in folders:
    path_base = folder_base + folder
    for file in files:
        for i, sub_folder in enumerate(sub_folders):
            src_path = path_base + sub_folder + file_base + file + '.pkl'
            dest_path = save_location_base + folder + '/' + file + '_' + str(i+1) + '.pkl'
            print(src_path, dest_path)
            copyfile(src_path, dest_path)