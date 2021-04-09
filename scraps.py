import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def graph_it():
    # print(big_roe(np.array([0, 3])))
    lin = np.arange(-5, 8, 0.1)
    func = np.sin(lin, 0.1)
    fig, ax = plt.subplots()
    ax.plot(lin, func)
    ax.grid()
    plt.show()


# All of this stuff is to solve the gambling problem recursively
p = 0.4722
q = 1 - p


def helper(f):
    return p / (1 - q * f)


def find_ratio():
    cur = p
    result = 1
    for i in range(1, 59):
        result *= cur
        cur = helper(cur)
    return result


def check_big_roe(big_roe):
    n = 10
    print(big_roe(np.ones([n, 1]) * 3))
    print((2 / 3) ** n)
    print(big_roe(np.ones([n, 1]) * 0))
    print((1 / 3) ** n)

    print("Hi there!")
    print(big_roe(np.append(np.ones([n, 1]) * 3, np.zeros([n, 1]))))
    print((2 / 3) ** n * (1 / 3) ** n)


# Finds the training and generalization error for some data, at each split
def deprecated_in_out_error(data, train, power=1, show=True, printout=False):
    # Keeps track of out in and out error for different sizes of training data
    mean_sq_err = np.zeros([3, len(train)])
    for i, number in enumerate(train):
        train, test = stand_n(data, number)
        # features and labels in the training and test data. Features in the training data are the design matrix
        design = train[:, :-1]
        train_lab = train[:, -1]
        test_feats = test[:, :-1]
        test_lab = test[:, -1]

        # Increasing the power of these polynomials
        design = add_powers(design, power)
        test_feats = add_powers(test_feats, power)

        # the transpose and inverse
        design_t = np.matrix.transpose(design)
        square = np.matmul(design_t, design)
        square_i = np.linalg.inv(square)
        coeff = np.matmul(square_i, design_t)
        weights = np.matmul(coeff, train_lab)
        # Finding the error for both the training and testing data
        in_err = estimate(design, train_lab, weights)
        out_err = estimate(test_feats, test_lab, weights)
        # Recording said errors
        mean_sq_err[0, i] = number
        mean_sq_err[1, i] = in_err
        mean_sq_err[2, i] = out_err
        # Printing out the weights, if specified
        if printout:
            print("\nWeights for values (working with " + str(number) + " data points)")
            for j in range(len(info_types) - 1):
                print(str(info_types[j]) + " has weight " + str(weights[j]))
                for pow in range(2, power + 1):
                    idx = j + (len(info_types) - 1) * (pow - 1)
                    print(str(info_types[j]) + "^" + str(pow) + " has weight " + str(weights[idx]))
    if show:
        plt.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="log-training error")
        plt.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="log-testing error")
        plt.legend()
        plt.ylabel("ln(Error)")
        plt.xlabel("Quantity of training data")
        plt.show()
    return mean_sq_err


# Runs the in_out_error function for multiple dimensions, and graphs the result
def deprecated_in_out_multi(data, splits, powers):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    for i, power in enumerate(powers):
        mean_sq_err = deprecated_in_out_error(data, splits, power=power, show=False)
        # Plotting the data
        alpha = 1
        if i < len(powers) / 2:
            alpha = max(0.1, -0.4 * (i + 1) + 1.4)
            alpha = min(1, alpha)
            ax1.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="degree: " + str(power), alpha=alpha, color='g')
            ax2.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="degree: " + str(power), alpha=alpha, color='r')
            ax1.plot(mean_sq_err[0], np.log(mean_sq_err[1]), alpha=alpha / 2, color='g')
            ax2.plot(mean_sq_err[0], np.log(mean_sq_err[2]), alpha=alpha / 2, color='r')
            # ax1.scatter(mean_sq_err[0, -2:], np.log(mean_sq_err[1, -2:]), label="degree: " + str(power), alpha=alpha, color='g')
            # ax2.scatter(mean_sq_err[0, -2:], np.log(mean_sq_err[2, -2:]), label="degree: " + str(power), alpha=alpha, color='r')
            # ax1.plot(mean_sq_err[0, -2:], np.log(mean_sq_err[1, -2:]), alpha=alpha / 2, color='g')
            # ax2.plot(mean_sq_err[0, -2:], np.log(mean_sq_err[2, -2:]), alpha=alpha / 2, color='r')
        else:
            alpha = min(1, 0.4 * (i + 1) - 1.4)
            alpha = max(0.1, alpha)
            ax1.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="degree: " + str(power), alpha=alpha, color='b')
            ax2.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="degree: " + str(power), alpha=alpha, color='purple')
            ax1.plot(mean_sq_err[0], np.log(mean_sq_err[1]), alpha=alpha / 2, color='b')
            ax2.plot(mean_sq_err[0], np.log(mean_sq_err[2]), alpha=alpha / 2, color='purple')
            # ax1.scatter(mean_sq_err[0, -2:], np.log(mean_sq_err[1, -2:]), label="degree: " + str(power), alpha=alpha, color='b')
            # ax2.scatter(mean_sq_err[0, -2:], np.log(mean_sq_err[2, -2:]), label="degree: " + str(power), alpha=alpha, color='purple')
            # ax1.plot(mean_sq_err[0, -2:], np.log(mean_sq_err[1, -2:]), alpha=alpha / 2, color='b')
            # ax2.plot(mean_sq_err[0, -2:], np.log(mean_sq_err[2, -2:]), alpha=alpha / 2, color='purple')
    ax1.legend()
    ax2.legend()
    ax1.set_title("log-training error")
    ax2.set_title("log-testing error")
    plt.ylabel("ln(Error)")
    plt.xlabel("Quantity of training data")
    plt.show()


if __name__ == "__main__":
    print(find_ratio())
