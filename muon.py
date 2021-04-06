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

# I want the data to be a global variable
raw_data = pd.read_table("lifetime.dat", sep="\s+")
lifetimes = np.array(raw_data["TIME"])
# Some general information about the data collected
# print(lifetimes)
# print("The sum of the lifetimes is " + str(np.sum(lifetimes)))
# print(np.average(lifetimes))


# Sampler based on the Metropolis Hastings (M-H) algorithm. It's a function that takes as inputs: the dimension d, the
# total number of steps in the Markov chain N, and the tuning parameter of the Gaussian proposal
# distribution beta > 0. When d > 0, we have a multi-dimensional Gaussian
def sampler(num_steps, beta):
    assert num_steps > 0 and beta > 0
    cur_tau = np.random.uniform(low=0.1, high=8)
    # This is the pdf evaluated at the current step
    old_prob = target_dist(cur_tau)
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
        # print("cur_tau is " + str(cur_tau))
        print(target_dist(cur_tau))
        # Prints out the time taken when it's 25%, 50%, and 75% finished running the algorithm
        # if i == num_steps // 4:
        #     print("25% complete (" + str(num_steps) + " steps, beta = " + str(beta) + ")")
        #     quarter_time = time.time()
        #     print("time taken in this quarter: " + str(quarter_time - mh_s))
        # elif i == num_steps // 2:
        #     print("50% complete (" + str(num_steps) + " steps, beta = " + str(beta) + ")")
        #     half_time = time.time()
        #     print("time taken in this quarter: " + str(half_time - quarter_time))
        # elif i == 3 * num_steps // 4:
        #     print("75% complete (" + str(num_steps) + " steps, beta = " + str(beta) + ")")
        #     print("time taken in this quarter: " + str(time.time() - half_time))
        # Draw a new step from the sampling distribution
        next_tau = prop_step(np.array([cur_tau]), beta)[0]
        # print("next_tau is " + str(next_tau))
        # The pdf evaluated at this proposed new step
        new_prob = target_dist(next_tau)
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
    axs.plot(lin, (delta * len(raw_data) / n_bins) * decay_func(lin, calculated_tau))
    if isinstance(title, str):
        plt.title(title)
    if show:
        plt.show()
    return counts


def target_dist(tau):
    # print("Input: " + str(tau) + " likelihood is: " + str(likelihood(tau)) + " prior is: " + str(prior(tau)))
    return log_likelihood(tau) + np.log(prior(tau))


# Likelihood function: probability of lifetime tau given the data in lifetime.dat
def log_likelihood(tau):
    array = 5.605 * decay_func(lifetimes, tau)
    print("array is " + str(array) + " with length " + str(len(array)) + " " + str(len(lifetimes) == len(array)))
    result = np.prod(np.log(array))
    print("product is " + str(result))
    return result


# Everything is measured in microseconds
def decay_func(t, tau):
    # Normalizing constant
    norm_const = 1 / (tau * (np.exp(-start_time / tau) - np.exp(-stop_time / tau)))
    return norm_const * np.exp(-t / tau)


# This is what I'm guessing the distribution is based on the frequentist analysis
def prior(tau):
    sigma = 1
    mean = 2.12
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    expo = - 1 / 2 * ((tau - mean) / sigma) ** 2
    return norm * np.exp(expo)


if __name__ == "__main__":
    target_dist(calculated_tau)
    target_dist(0.1)
    # journey, accept_rate = sampler(1, 0.3)
    # counts = histogram(journey, show=True)
