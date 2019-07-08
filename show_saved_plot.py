import pickle
from matplotlib import pyplot as plt
import os
import sys


def show_saved_plots(plot_location):
    print("Showing all {} plots in directory {}".format(len(os.listdir(plot_location)), plot_location))
    for filename in os.listdir(plot_location):
        with open(plot_location + filename, 'rb') as file:
            fig = pickle.load(file)
    plt.show()


def show_saved_plot(plot_location):
    print("Showing {}".format(plot_location))
    with open(plot_location, 'rb') as file:
        fig = pickle.load(file)
    plt.show()

if __name__ == '__main__':
    plot_location = sys.argv[1]
    if plot_location[-4:] == '.pkl':
        show_saved_plot(plot_location)
    else:
        show_saved_plots(plot_location)