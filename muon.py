import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

n_bins = 50
start_time = 0.1
stop_time = 8

calculated_tau = 2.12

# Everything is measured in microseconds
def decay_func(t, tau):
    # Normalizing constant
    norm_const = 1 / (tau * (np.exp(-start_time / tau) - np.exp(-stop_time / tau)))
    return norm_const * np.exp(-t / tau)


def histogram(lab_data, title=None, show=True):
    assert isinstance(lab_data, np.ndarray)
    # n_bins = min(len(lab_data) // 200, 100)
    # if n_bins < 20:
    #     n_bins = 20
    fig, axs = plt.subplots(1)
    # We can set the number of bins with the `bins` kwarg
    counts, bins, bars = axs.hist(lab_data, bins=n_bins)
    # Plotting the proposal distribution
    start = -2
    stop = 5
    delta = stop - start
    lin = np.linspace(start_time, stop_time)
    axs.plot(lin, (delta * len(data) / n_bins) * decay_func(lin, calculated_tau))
    if isinstance(title, str):
        plt.title(title)
    if show:
        plt.show()
    return counts


data = pd.read_table("lifetime.dat", sep="\s+")

# print(data)
lifetimes = np.array(data["TIME"])
print(lifetimes)
print("The sum of the lifetimes is " + str(np.sum(lifetimes)))
print("The square root of that is " + str(np.sqrt(np.sum(lifetimes))))
print(np.average(lifetimes))
# Creates a histogram out of the data
counts = histogram(lifetimes, show=True)

# lab_data = np.zeros([50, 2])
# for i, element in enumerate(counts):
#     increment = (8 - 0.1) / 50
#     lab_data[i, 0] = start_time + i * increment
#     lab_data[i, 1] = element
# np.savetxt("histogram.csv", lab_data, delimiter=",")
