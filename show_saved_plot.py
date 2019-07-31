import pickle
import matplotlib
from matplotlib import pyplot as plt
import os
import sys
import numpy as np


def show_saved_plots(plot_location):
    print("Showing all {} plots in directory {}".format(len(os.listdir(plot_location)), plot_location))
    for filename in os.listdir(plot_location):
        with open(plot_location + filename, 'rb') as file:
            fig = pickle.load(file)
            fig.canvas.set_window_title(filename)

            ax = fig.axes[0]
            line = ax.lines[0]
            xs = np.array(line.get_xdata())
            ys = np.array(line.get_ydata())

            min_x = 500 * 500000 + 1000000 / 2
            max_x = min_x + 1000000
            mean_test_value = np.mean(ys[np.bitwise_and(xs > min_x, (xs <= max_x))])
            print("{}: {}".format(filename, mean_test_value))

    plt.show()


def show_saved_plot(plot_location):
    print("Showing {}".format(plot_location))
    with open(plot_location, 'rb') as file:
        fig = pickle.load(file)
    plt.show()

if __name__ == '__main__':
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    plot_location = sys.argv[1]
    if plot_location[-4:] == '.pkl':
        show_saved_plot(plot_location)
    else:
        show_saved_plots(plot_location)
