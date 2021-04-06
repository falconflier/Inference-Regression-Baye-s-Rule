import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mcmc_dist import prop_step

n_bins = 50
start_time = 0.1
stop_time = 8

calculated_tau = 2.12
# This is the range from which we calculate the probability distribution
term = 1000


# Sampler based on the Metropolis Hastings (M-H) algorithm. It's a function that takes as inputs: the dimension d, the
# total number of steps in the Markov chain N, and the tuning parameter of the Gaussian proposal
# distribution beta > 0. When d > 0, we have a multi-dimensional Gaussian
def sampler(dim, num_steps, beta):
    assert dim > 0 and num_steps > 0 and beta > 0
    # The initial position in d-space
    # cur_pos = np.zeros([dim])
    # cur_pos = np.random.uniform(-1, 4, [dim])
    cur_pos = np.zeros([dim])
    # This is the pdf evaluated at the current step
    old_prob = decay_func(cur_pos)
    # Keeps track of all past steps
    journey = np.zeros([num_steps])
    # Keeps track of the number of accepted proposals in the last 10000 steps
    num_accepts = 0
    # Activates when we're in the last 10000 steps
    is_ending = False
    mh_s = time.time()
    quarter_time = 0
    half_time = 0

    everything = np.zeros(num_steps)
    # Iterates R&R for the specified number of steps
    for i in range(num_steps):
        # Record where we've gone (either we accepted the proposal, or we stayed put)
        journey[i] = cur_pos[0]
        # Prints out the time taken when it's 25%, 50%, and 75% finished running the algorithm
        if i == num_steps // 4:
            print("25% complete (" + str(dim) + "-dimensions, " + str(num_steps) + " step, beta = " + str(beta) + ")")
            quarter_time = time.time()
            print("time taken in this quarter: " + str(quarter_time - mh_s))
        elif i == num_steps // 2:
            print("50% complete (" + str(dim) + "-dimensions, " + str(num_steps) + " step, beta = " + str(beta) + ")")
            half_time = time.time()
            print("time taken in this quarter: " + str(half_time - quarter_time))
        elif i == 3 * num_steps // 4:
            print("75% complete (" + str(dim) + "-dimensions, " + str(num_steps) + " step, beta = " + str(beta) + ")")
            print("time taken in this quarter: " + str(time.time() - half_time))
        # Draw a new step from the sampling distribution
        next_pos = prop_step(cur_pos, beta)
        # The pdf evaluated at this proposed new step
        new_prob = decay_func(next_pos)
        # If the new position has higher probability OR we draw a number less than their ratios, accept the proposal
        rand_number = np.random.random()
        if new_prob > old_prob or rand_number < (new_prob / old_prob):
            # print("new step is " + str(next_pos))
            # print("took that step with probability " + str(big_roe(journey[i]) / big_roe(journey[i - 1])))
            # print("random number was " + str(rand_number))
            # Update the current position
            cur_pos = next_pos
            # Update the old probability
            old_prob = new_prob
            # This proposal was accepted, so we increment the number of acceptances
            if is_ending or i > num_steps - term:
                is_ending = True
                num_accepts += 1
    mh_e = time.time()
    print("It took " + str(mh_e - mh_s) + " seconds to run MH algorithm")
    accept_rate = -1
    if num_steps > term:
        print("number of accepts was " + str(num_accepts))
        accept_rate = num_accepts / term
        print("The percentage of accepted moves in the last " + str(term) + " steps is " + str(accept_rate))

    return journey, accept_rate


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


# Everything is measured in microseconds
def decay_func(t, tau):
    # Normalizing constant
    norm_const = 1 / (tau * (np.exp(-start_time / tau) - np.exp(-stop_time / tau)))
    return norm_const * np.exp(-t / tau)


# This is what I'm guessing the distribution is based on the frequentist analysis
def prior(x):
    sigma = 1
    mean = 2.12
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    expo = - 1 / 2 * ((x - mean) / sigma) ** 2
    return norm * np.exp(expo)


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
