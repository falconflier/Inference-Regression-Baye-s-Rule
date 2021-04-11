import time
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# We often only care about the last 10000 and I wanted to prevent magic numbers
term = 10000


# Raises 10 to the specified power
def power(exp):
    return 10 ** exp


# defines the one-dimensional probability distribution specified in the handout. Multidimensional distributions will
# be constructed by the product of these "simple" one dimensional distributions
def roe(x):
    assert isinstance(x, (int, float, np.integer, np.floating, np.ndarray))
    return (1 / 3) * (np.exp(-x ** 2) + 2 * np.exp(-(x - 3) ** 2))


# The multidimensional distribution. Takes in an array of d values
def big_roe(d_vector):
    # assert isinstance(d_vector, np.ndarray)
    # assert np.all(isinstance(, (np.floating, float, int)))
    return np.prod(roe(d_vector))
    # result = 1
    # for element in d_vector:
    #     # print("element is " + str(element) + " with type " + str(type(element)))
    #     result *= roe(element)
    # return result


# Uses a multidimensional gaussian to propose a new position for the distribution
def prop_step(old_pos, beta):
    assert isinstance(old_pos, np.ndarray)
    assert isinstance(beta, (int, np.integer, float, np.floating))
    assert beta > 0
    # finding the standard deviation so I can plug it directly into the builtin function random provides
    # sigma = 1 / (2 * beta)
    sigma = beta
    new_pos = np.zeros([len(old_pos)])
    for i, element in enumerate(old_pos):
        new_pos[i] = element + np.random.normal(scale=sigma)
    return new_pos


# Creates a histogram out of the data
def histogram(data, title=None, save=None):
    assert isinstance(data, np.ndarray)
    n_bins = min(len(data) // 200, 100)
    if n_bins < 20:
        n_bins = 20
    fig, axs = plt.subplots(1)
    # We can set the number of bins with the `bins` kwarg
    axs.hist(data, bins=n_bins)
    # Plotting the proposal distribution
    start = -2
    stop = 5
    delta = stop - start
    lin = np.linspace(start, stop)
    axs.plot(lin, (delta * len(data) / n_bins) * roe(lin))
    if isinstance(save, str):
        plt.title(title)
    if save:
        plt.savefig(save + " histogram")
        plt.close()
    else:
        plt.show()


# Takes the elements of a numpy array and creates a trace plot.
def trace_plot(matrix, show_hist=True, title=None, save=None):
    assert isinstance(matrix, np.ndarray)
    # aggregate = np.array([])
    # for i, element in np.ndenumerate(matrix):
    #     # print("index is " + str(i) + " and element is " + str(element))
    #     # Records the first dimension's value if d > 1
    #     if isinstance(element, np.ndarray):
    #         aggregate = np.append(aggregate, element[0])
    #     # Records the parameter for the case that d == 1
    #     else:
    #         aggregate = np.append(aggregate, element)
    aggregate = matrix
    if matrix.ndim > 1:
        aggregate = matrix[0, :]
    x_ax = np.arange(0, len(aggregate))
    plt.plot(x_ax, aggregate)

    if isinstance(title, str):
        plt.title(title)

    # if we've been told to save, we'll save it to the specified directory rather than displaying it
    if isinstance(save, str):
        plt.savefig(save + " trace plot")
        plt.close()
    else:
        plt.show()

    # Runs the histogram plotting method if asked to
    if show_hist:
        histogram(aggregate, title, save=save)


# Sampler based on the Metropolis Hastings (M-H) algorithm. It's a function that takes as inputs: the dimension d, the
# total number of steps in the Markov chain N, and the tuning parameter of the Gaussian proposal
# distribution beta > 0. When d > 0, we have a multi-dimensional Gaussian
def sampler(dim, num_steps, beta, verbose=True):
    assert dim > 0 and num_steps > 0 and beta > 0
    # The initial position in d-space
    # cur_pos = np.zeros([dim])
    cur_pos = np.random.uniform(-2, 5, [dim])
    # This is the pdf evaluated at the current step
    old_prob = big_roe(cur_pos)
    # Keeps track of all past steps
    journey = np.zeros([num_steps])
    # Keeps track of the number of accepted proposals in the last 10000 steps
    num_accepts = 0
    # Activates when we're in the last 10000 steps
    is_ending = False
    mh_s = time.time()
    quarter_time = 0
    half_time = 0

    # Iterates R&R for the specified number of steps
    for i in range(num_steps):
        # Record where we've gone (either we accepted the proposal, or we stayed put)
        journey[i] = cur_pos[0]
        if verbose:
            if i == num_steps // 4:
                print(
                    "25% complete (" + str(dim) + "-dimensions, " + str(num_steps) + " step, beta = " + str(beta) + ")")
                quarter_time = time.time()
                print("time taken in this quarter: " + str(quarter_time - mh_s))
            elif i == num_steps // 2:
                print(
                    "50% complete (" + str(dim) + "-dimensions, " + str(num_steps) + " step, beta = " + str(beta) + ")")
                half_time = time.time()
                print("time taken in this quarter: " + str(half_time - quarter_time))
            elif i == 3 * num_steps // 4:
                print(
                    "75% complete (" + str(dim) + "-dimensions, " + str(num_steps) + " step, beta = " + str(beta) + ")")
                print("time taken in this quarter: " + str(time.time() - half_time))
        # Draw a new step from the sampling distribution
        next_pos = prop_step(cur_pos, beta)
        # The pdf evaluated at this proposed new step
        new_prob = big_roe(next_pos)
        # If the new position has higher probability OR we draw a number less than their ratios, accept the proposal
        if new_prob > old_prob or np.random.random() < (new_prob / old_prob):
            # Update the current position
            cur_pos = next_pos
            # Update the old probability
            old_prob = new_prob
            # This proposal was accepted, so we decrement the number of rejects
            if is_ending or i > num_steps - term:
                is_ending = True
                num_accepts += 1
    mh_e = time.time()
    if verbose:
        print("It took " + str(mh_e - mh_s) + " seconds to run MH algorithm")
    accept_rate = -1
    if num_steps > term:
        accept_rate = num_accepts / term
        if verbose:
            print("The percentage of accepted moves in the last " + str(term) + " steps is " + str(accept_rate))
    return journey, accept_rate


# Runs the sampler for a given beta and dimension. Returns the acceptance rate
def dim_plot(dimension, beta=0.3, generate_images=True, verbose=True):
    # Beta is inversely proportional to the width of the spread. Large beta implies thin gaussians; small beta implies
    # lots of variance
    steps = power(6)
    path, rate = sampler(dimension, steps, beta, verbose=verbose)
    if generate_images:
        title = str(dimension) + "-dimensions, " + str(steps) + " steps, beta = " + str(
            beta) + ", acceptance rate = " + str(rate)
        # directory to save images of graphs to
        save_dir = "./graphs/" + str(dimension) + "d " + str(steps) + "n"
        trace_s = time.time()
        trace_plot(path, show_hist=True, title=title, save=save_dir)
        trace_e = time.time()
        print("It took " + str(trace_e - trace_s) + " seconds to run trace_plot in " + str(dimension) + " dimensions")
    return rate


# Uses the dim_plot method to find a lower bound on beta (trying to get acceptance_rate ~ 23.4%
# uses the dimensions start and stop (inclusive) if specified
def get_low_bound(start=8, stop=20):
    lower_bound = np.zeros([stop - start + 1, 3])
    for i in range(start, stop + 1):
        beta = 0.01
        accept_rate = -1
        while True:
            lower_bound[i - start] = np.array([i, beta, accept_rate])
            accept_rate = dim_plot(i, beta, generate_images=False)
            if accept_rate < 0.234:
                if accept_rate < 0.1:
                    beta += 0.2
                elif accept_rate < 0.15:
                    beta += 0.1
                else:
                    beta += 0.02
            else:
                break
    print("lower bounds are:")
    for row in lower_bound:
        string = ""
        for element in row:
            string = string + str(element) + ", "
        print(string)
    return lower_bound


# Uses the dim_plot method to find a lower bound on beta (trying to get acceptance_rate ~ 23.4%
# Uses the start and stop dimensions found in the lower_bound array
def get_upper_bound(lower_bound):
    start = int(lower_bound[0, 0])
    stop = int(lower_bound[-1, 0])
    print("start is " + str(start))
    print("stop is " + str(stop))
    upper_bound = np.zeros([stop - start + 1, 3])
    for i in range(start, stop + 1):
        beta = lower_bound[i - start, 1] + 0.5
        accept_rate = -1
        while True:
            upper_bound[i - start] = np.array([i, beta, accept_rate])
            accept_rate = dim_plot(i, beta, generate_images=False)
            if accept_rate > 0.234:
                if accept_rate > 0.5:
                    beta -= 0.2
                elif accept_rate > 0.35:
                    beta -= 0.1
                else:
                    beta -= 0.02
            else:
                break
    print("upper bounds are:")
    for row in upper_bound:
        string = ""
        for element in row:
            string = string + str(element) + ", "
        print(string)
    return upper_bound


# Gets trace plots for many dimensions
def many_dims(adjusts=False):
    # If we want to use a beta that varies based on dimension
    if adjusts:
        for i in range(1, 6):
            dim_plot(i, beta=beta_dependence(i))
        for i in np.arange(70, 101, 10):
            dim_plot(i, beta=beta_dependence(i))
    # If we want to use a beta that
    else:
        for i in range(1, 6):
            dim_plot(i)
        for i in np.arange(70, 101, 10):
            dim_plot(i)


# Encodes the dimensionality dependence of beta, beta -> beta(dim). Found empirically
def beta_dependence(dim):
    return 0.223 + (4.579 / dim)


def finding_beta():
    print("5 dimensional plot has acceptance rate: " + str(dim_plot(5, beta=0.3, generate_images=False, verbose=False)))

    # The new plots
    acceptance_rates = []

    beta = 0.80 #
    acceptance_rates.append([beta, dim_plot(8, beta=beta, generate_images=False, verbose=False)])
    print("8 dimensional plot has acceptance rate: " + str(acceptance_rates[7 - 7]))

    beta = 0.72 #
    acceptance_rates.append([beta, dim_plot(9, beta=beta, generate_images=False, verbose=False)])
    print("9 dimensional plot has acceptance rate: " + str(acceptance_rates[8 - 7]))

    beta = 0.68 #
    acceptance_rates.append([beta, dim_plot(10, beta=beta, generate_images=False, verbose=False)])
    print("10 dimensional plot has acceptance rate: " + str(acceptance_rates[9 - 7]))

    beta = 0.64 #
    acceptance_rates.append([beta, dim_plot(11, beta=beta, generate_images=False, verbose=False)])
    print("11 dimensional plot has acceptance rate: " + str(acceptance_rates[10 - 7]))

    beta = 0.61 #
    acceptance_rates.append([beta, dim_plot(12, beta=beta, generate_images=False, verbose=False)])
    print("12 dimensional plot has acceptance rate: " + str(acceptance_rates[11 - 7]))

    beta = 0.58 #
    acceptance_rates.append([beta, dim_plot(13, beta=beta, generate_images=False, verbose=False)])
    print("13 dimensional plot has acceptance rate: " + str(acceptance_rates[12 - 7]))

    beta = 0.56 ##
    acceptance_rates.append([beta, dim_plot(14, beta=beta, generate_images=False, verbose=False)])
    print("14 dimensional plot has acceptance rate: " + str(acceptance_rates[13 - 7]))

    beta = 0.53 #<
    acceptance_rates.append([beta, dim_plot(15, beta=beta, generate_images=False, verbose=False)])
    print("15 dimensional plot has acceptance rate: " + str(acceptance_rates[14 - 7]))

    beta = 0.51 #>
    acceptance_rates.append([beta, dim_plot(16, beta=beta, generate_images=False, verbose=False)])
    print("16 dimensional plot has acceptance rate: " + str(acceptance_rates[15 - 7]))

    beta = 0.49 #
    acceptance_rates.append([beta, dim_plot(17, beta=beta, generate_images=False, verbose=False)])
    print("17 dimensional plot has acceptance rate: " + str(acceptance_rates[16 - 7]))

    beta = 0.475
    acceptance_rates.append([beta, dim_plot(18, beta=beta, generate_images=False, verbose=False)])
    print("18 dimensional plot has acceptance rate: " + str(acceptance_rates[17 - 7]))

    beta = 0.46
    acceptance_rates.append([beta, dim_plot(19, beta=beta, generate_images=False, verbose=False)])
    print("19 dimensional plot has acceptance rate: " + str(acceptance_rates[18 - 7]))

    beta = 0.45
    acceptance_rates.append([beta, dim_plot(20, beta=beta, generate_images=False, verbose=False)])
    print("20 dimensional plot has acceptance rate: " + str(acceptance_rates[19 - 7]))
    # writing to a csv
    f = open('acceptance_info.csv', 'a')
    # test = [1, 2, 3, 4, 5, 6]
    # test2 = [8, 7, 6, 5]
    line = ""
    for element in acceptance_rates:
        line = line + str(element) + ","
    f.write('\n')
    f.write(line)
    f.close()


if __name__ == "__main__":
    many_dims(adjusts=True)
    # low_bound = get_low_bound(start=8, stop=20)
    # upper_bound = get_upper_bound(low_bound)
    # upper_bound = get_upper_bound(np.array([[8, 0.1], [-100, 0.2], [-100, 0.3], [-100, 0.4], [12, 0.5]]))
    # for i in range(8, 21):
    #     print("for dimension " + str(i) + " lower bound on beta is " + str(low_bound[i, 1]) + " and upper bound is " + str(upper_bound[i, 1]) + " acceptance rates of " + str(low_bound[i, 2]) + " and " + str(upper_bound[i, 2]))
