from figure_creation.figure_helpers import load_object, get_smoothed_data
import numpy as np

folder_base = 'D:/MSc Plots/Deltic Plots/experiment_results/target_network_lrx100_'

folders = ['noiseless', 'noise0-15', 'noise0-30', 'noise0-45', 'noise0-60', 'noise0-90', 'noise1-20']
#folders = ['noise1-20']

#sub_folders = ['/test_500', '/test_750', '/test_1000', '/test_1250']
sub_folders = ['/test_500', '/test_750', '/test_1000']

file_base = '/monitors/'

save_location_base = 'C:/Users/Will/PycharmProjects/MScProject/figure_creation/target_network_noise/raw_data/'

train_errors = []
test_errors = []
for folder in folders:
    path_base = folder_base + folder
    best_test_error = np.inf
    best_train_error = np.inf
    for i, sub_folder in enumerate(sub_folders):
        src_path = path_base + sub_folder + file_base + 'sum_squares_error.pkl'
        data = load_object(src_path)[1]
        test_error = np.mean(data[-1000:])

        smoothed_error = get_smoothed_data(data, 35000)
        min_training_error = np.min(smoothed_error)
        best_train_error = min(min_training_error, best_train_error)
        best_test_error = min(best_test_error, test_error)

        print(test_error, min_training_error, folder, sub_folder)

    print(best_test_error, folder, 'best')
    print()
    train_errors += [best_train_error]
    test_errors += [best_test_error]



import matplotlib.pyplot as plt

#plt.plot([0, 0.15, 0.30, 0.45, 0.6, 0.9, 1.2], [0.09022911465989959, 0.08967617669387362, 0.08861968903256999, 0.08699136676083595, 0.08551288832369616, 0.13205351973933888, 0.12959720375218972])
plt.figure()
plt.plot([0, 0.15, 0.30, 0.45, 0.6, 0.9, 1.2], train_errors)
plt.figure()
plt.plot([0, 0.15, 0.30, 0.45, 0.6, 0.9, 1.2], test_errors)
#plt.ylim((0, max(test_errors) + 0.01))
plt.show()

