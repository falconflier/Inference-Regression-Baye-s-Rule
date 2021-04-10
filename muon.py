import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mcmc_dist import prop_step
from scipy.stats import norm
from scipy.optimize import minimize

n_bins = 100
start_time = 0.1
stop_time = 8
calculated_tau = 2.436
# This is the point at which we begin calculating acceptance rate
term = 1000

# I want the data to be a global variable
raw_data = pd.read_table("lifetime.dat", sep="\s+")
lifetimes = np.array(raw_data["TIME"])
# Some general information about the data collected
# print(lifetimes)
# print("The sum of the lifetimes is " + str(np.sum(lifetimes)) + ", length is " + str(len(lifetimes)))
# print(np.mean(lifetimes))


# Sampler based on the Metropolis Hastings (M-H) algorithm. It's a function that takes as inputs: the dimension d, the
# total number of steps in the Markov chain N, and the tuning parameter of the Gaussian proposal
# distribution beta > 0. When d > 0, we have a multi-dimensional Gaussian
# Beta is the std_dev for the proposal distribution, sigma is the std_dev for the target distribution
def sampler(num_steps, beta, sigma):
    assert num_steps > 0 and beta > 0
    cur_tau = calculated_tau
    # This is the pdf evaluated at the current step
    old_prob = target_dist(cur_tau, sigma)
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
        journey[i] = cur_tau
        # Draw a new step from the sampling distribution
        next_tau = prop_step(np.array([cur_tau]), beta)[0]
        # print("next_tau is " + str(next_tau))
        # The pdf evaluated at this proposed new step
        new_prob = target_dist(next_tau, sigma)
        # If the new position has higher probability OR we draw a number less than their ratios, accept the proposal
        rand_number = np.random.random()
        # print("new prob is " + str(new_prob) + " old prob is " + str(old_prob) + " rand number is " + str(rand_number))
        if new_prob > old_prob or rand_number < (new_prob / old_prob):
            # Update the current position
            cur_tau = next_tau
            # Update the old probability
            old_prob = new_prob
            # This proposal was accepted, so we increment the number of acceptances
            if is_ending or i > num_steps - term:
                is_ending = True
                num_accepts += 1
    accept_rate = -1
    # Printing out acceptance rates
    if num_steps > term:
        print("number of accepts was " + str(num_accepts))
        accept_rate = num_accepts / term
        print("The percentage of accepted moves in the last " + str(term) + " steps is " + str(accept_rate))

    return journey, accept_rate


def histogram(input_data, sigma, show=True):
    assert isinstance(input_data, np.ndarray)
    fig, axs = plt.subplots(1)
    # finding the confidence interval and median
    object_from_scipy = norm(*norm.fit(input_data))
    ci = object_from_scipy.interval(0.68)
    median = object_from_scipy.median()
    # We can set the number of bins with the `bins` kwarg
    counts, bins, bars = axs.hist(input_data, bins=n_bins, zorder=1)
    # Plotting the frequentist distribution
    start = 1.7
    stop = 3
    lin = np.linspace(start, stop)
    max_height = np.amax(counts)
    # Plotting the median
    plt.vlines(median, 0, max_height + 100, colors="purple", linestyles=':', zorder=15)
    # Showing the confidence interval on the plot
    plt.fill_betweenx([0, max_height + 100], ci[0], ci[1], color='orange', alpha=0.2, zorder=5)
    print("confidence interval is " + str(ci[0]) + " to " + str(ci[1]))
    axs.plot(lin, max_height * freq_dist(lin, sigma) * sigma, color='r', zorder=10, alpha=0.5)
    # axs.plot(lin, max_height * likelihood(lin))
    title = "Muon lifetime median: " + str(np.round(median, decimals=3)) + " CI is: " + str(np.around(ci, decimals=3))
    plt.title(title)
    if show:
        plt.show()
    return counts


def target_dist(tau, sigma):
    # print("Input: " + str(tau) + " likelihood is: " + str(likelihood(tau)) + " prior is: " + str(prior(tau)))
    return likelihood(tau) * prior(tau, sigma)


# Likelihood function: probability of a lifetime tau given the data in lifetime.dat
def likelihood(tau):
    array = 5.605 * decay_func(lifetimes, tau)
    # array = decay_func(lifetimes, tau)
    # print("array is " + str(array) + " with length " + str(len(array)) + " " + str(len(lifetimes) == len(array)))
    # print("product is " + str(np.prod(array)))
    return np.prod(array)


# Everything is measured in microseconds
def decay_func(t, tau):
    # Normalizing constant
    norm_const = 1 / (tau * (np.exp(-start_time / tau) - np.exp(-stop_time / tau)))
    return norm_const * np.exp(-t / tau)


def freq_dist(tau, sigma):
    # This is what I'm guessing the distribution is based on the frequentist analysis
    mean = calculated_tau
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    expo = - 1 / 2 * ((tau - mean) / sigma) ** 2
    return norm * np.exp(expo)


def prior(tau, sigma):
    # A "uniform" distribution over a guess. Scaled by a constant factor, but that shouldn't matter
    if isinstance(tau, (np.floating, float)):
        if 1.5 < tau < 3.5:
            return 1
        else:
            return 0
    for i, time in enumerate(tau):
        if 1.5 < time < 3.5:
            tau[i] = 1
        else:
            tau[i] = 0
    return tau


# Useful for optimizations (because we can minimize)
def neg_likelihood(tau):
    return - likelihood(tau)


def show_freq_best(tau):
    fig, axs = plt.subplots(1)
    # We can set the number of bins with the `bins` kwarg
    counts, bins, bars = axs.hist(lifetimes, bins=n_bins, zorder=1)
    # Plotting the frequentist distribution
    start = 0.1
    stop = 8.0
    lin = np.linspace(start, stop)
    max_height = np.amax(counts)
    axs.plot(lin, max_height * decay_func(lin, tau), color='r', zorder=10, alpha=1)
    title = "Best fit from frequentist analysis"
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    show_freq_best(calculated_tau)

    opt_tau = minimize(neg_likelihood, 2.2)
    print(opt_tau)
    journey, accept_rate = sampler(10 ** 6, 0.25, 1)
    counts = histogram(journey, 0.1, show=True)
