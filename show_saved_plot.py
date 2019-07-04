import pickle
from matplotlib import pyplot as plt
import os

experiment_location = 'experiment_plots/target_network/test/plot_objects/'

#filename = 'feedback_interneuron_angle_07-07PM July04.pkl'

print(len(os.listdir(experiment_location)))
for filename in os.listdir(experiment_location):
    with open(experiment_location + filename, 'rb') as file:
        fig = pickle.load(file)

plt.show()
