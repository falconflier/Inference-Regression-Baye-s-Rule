import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge


info_types = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


# Used to plot individual features to see what impacts MEDV most directly
def make_plot(data):
    med_value = np.array(data["MEDV"])
    plt.ylabel("Median Value ($1000s)")
    column = "LSTAT"
    plt.xlabel(" % lower status of the population")
    x = np.array(data[column])
    plt.scatter(x, med_value)
    # plt.savefig("housing/" + column)
    # plt.show()


# Finding the best of the labels using the given weights, and finding the mean squared error
def estimate(design, labels, weights):
    est = np.matmul(design, weights)
    dif = labels - est
    dif_sq = np.dot(dif, dif)
    # mean squared error
    err = np.sum(dif_sq) / len(labels)
    return err


# Takes in a non-standardized array of data, and adds higher powers of the features (ignores the last column)
def add_powers(array, power):
    assert power > 0
    # This gets flipped, and we really need an n by 1 matrix
    labels = np.reshape(array[:, -1], [len(array), 1])
    array = array[:, :-1]
    result = array
    ones = np.ones([len(result), len(result[0])])
    for i in range(2, power + 1):
        to_add = array ** (ones * i)
        result = np.hstack((result, to_add))
    result = np.hstack((result, labels))
    return result


# standardizes the first n elements by subtracting the mean and dividing by the standard
# deviation. Then standardizes the remaining (test) data using the same mean and standard deviation
def stand_n(array, n, ignore=None):
    assert n > 1
    result = np.transpose(array)
    for i in range(len(array[0])):
        # ignore allows us to skip some data types (e.g. if we have binary data types)
        if isinstance(ignore, list):
            if i in list:
                continue
        mean = np.mean(result[i, :n])
        std_dev = np.std(result[i, :n])
        # this is to avoid divide-by-zero errors. Happens when all the data is exactly the same (only in code-testing)
        if std_dev == 0:
            std_dev = 1
        result[i] = (result[i] - mean) / std_dev
    return np.transpose(result)


# Adds powers and standardizes a very simple array
def test_add_powers():
    test = np.reshape(np.arange(1, 10), [3, 3])
    test = np.hstack((test, np.ones([3, 1]) * (-1)))
    print(test)
    longer = add_powers(test, 3)
    print("Add powers gave me\n" + str(longer))
    print("Standardizing the array gives me\n" + str(stand_n(longer, 3).round(3)))
    train, test = split_n(test, 2)
    # print("Splitting gave \n" + str(test) + "\nand testing data \n" + str(test))


# Splits the given array in two for training and test data.
def split_n(array, n):
    # print("Got array \n" + str(array))
    training = array[:n, :]
    test = array[n:, :]
    # print("Returning arrays \n" + str(training) + "\n and \n" + str(test))
    return training, test


# Finds the training and generalization error for some data, at each split. Takes in standardized data, wrote all the
# stuff pretty much by myself
def homemade_in_out_error(input_data, splits, show=True, printout=False):
    # Keeps track of out in and out error for different sizes of training data
    mean_sq_err = np.zeros([3, len(splits)])
    for i, number in enumerate(splits):
        # Standardizing the data
        input_data = stand_n(input_data, number)
        # Splitting it into training and testing data
        train, test = split_n(input_data, number)
        # features and labels in the training and test data. Features in the training data are the design matrix
        design = train[:, :-1]
        train_lab = train[:, -1]
        test_feats = test[:, :-1]
        test_lab = test[:, -1]

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
            power = int((len(input_data[0]) - 1) / 13)
            print("\nWeights for values (working with " + str(number) + " data points)")
            for j in range(len(info_types) - 1):
                print(str(info_types[j]) + " has weight " + str(weights[j]))
                for pow in range(2, power + 1):
                    idx = j + (len(info_types) - 1) * (pow - 1)
                    print(str(info_types[j]) + "^" + str(pow) + " has weight " + str(weights[idx]))
    if show:
        plt.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="log-training error")
        plt.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="log-testing error")
        # plt.scatter(mean_sq_err[0], mean_sq_err[1], label="training error")
        # plt.scatter(mean_sq_err[0], mean_sq_err[2], label="testing error")
        plt.legend()
        plt.ylabel("ln(Error)")
        plt.xlabel("Quantity of training data")
        plt.show()
    return mean_sq_err


# Finds the training and generalization error for some data, at each split. Takes in standardized data, uses sklearn
def in_out_error(input_data, splits, show=True, printout=False):
    # Keeps track of out in and out error for different sizes of training data
    mean_sq_err = np.zeros([3, len(splits)])
    for i, number in enumerate(splits):
        # Standardizing the data
        input_data = stand_n(input_data, number)
        # Splitting it into training and testing data
        train, test = split_n(input_data, number)
        # features and labels in the training and test data. Features in the training data are the design matrix
        X = train[:, :-1]
        y = train[:, -1]
        test_feats = test[:, :-1]
        test_lab = test[:, -1]

        reg = LinearRegression().fit(X, y)
        weights = reg.coef_

        # Finding the error for both the training and testing data
        in_err = estimate(X, y, weights)
        out_err = estimate(test_feats, test_lab, weights)
        # Recording said errors
        mean_sq_err[0, i] = number
        mean_sq_err[1, i] = in_err
        mean_sq_err[2, i] = out_err
        # Printing out the weights, if specified
        if printout:
            power = int((len(input_data[0]) - 1) / 13)
            print("\nWeights for values (working with " + str(number) + " data points)")
            for j in range(len(info_types) - 1):
                print(str(info_types[j]) + " has weight " + str(weights[j]))
                for pow in range(2, power + 1):
                    idx = j + (len(info_types) - 1) * (pow - 1)
                    print(str(info_types[j]) + "^" + str(pow) + " has weight " + str(weights[idx]))
    if show:
        plt.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="log-training error")
        plt.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="log-testing error")
        # plt.scatter(mean_sq_err[0], mean_sq_err[1], label="training error")
        # plt.scatter(mean_sq_err[0], mean_sq_err[2], label="testing error")
        plt.legend()
        plt.ylabel("ln(Error)")
        plt.xlabel("Quantity of training data")
        plt.show()
    return mean_sq_err


# Runs the in_out_error function for multiple dimensions, and graphs the result
def in_out_multi(input_data, splits, max_pow):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    for i, power in enumerate(max_pow):
        print("Now working on power " + str(power))
        poly_data = add_powers(input_data, power)
        mean_sq_err = in_out_error(poly_data, splits, show=False)
        # Plotting the data
        alpha = 1
        if i < len(max_pow) / 2:
            alpha = max(0.1, -0.4 * (i + 1) + 1.4)
            alpha = min(1.0, alpha)
            ax1.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="degree: " + str(power), alpha=alpha, color='g')
            ax2.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="degree: " + str(power), alpha=alpha, color='r')
            ax1.plot(mean_sq_err[0], np.log(mean_sq_err[1]), alpha=alpha / 2, color='g')
            ax2.plot(mean_sq_err[0], np.log(mean_sq_err[2]), alpha=alpha / 2, color='r')
        else:
            alpha = min(1.0, 0.4 * (i + 1) - 1.4)
            alpha = max(0.1, alpha)
            ax1.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="degree: " + str(power), alpha=alpha, color='b')
            ax2.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="degree: " + str(power), alpha=alpha, color='purple')
            ax1.plot(mean_sq_err[0], np.log(mean_sq_err[1]), alpha=alpha / 2, color='b')
            ax2.plot(mean_sq_err[0], np.log(mean_sq_err[2]), alpha=alpha / 2, color='purple')
    ax1.legend()
    ax2.legend()
    ax1.set_title("log-training error")
    ax2.set_title("log-testing error")
    plt.ylabel("ln(Error)")
    plt.xlabel("Quantity of training data")
    plt.show()


# Does ridge regression on the data for different lamba coefficients, and plots the results
def multi_ridge_reg(input_data, lambdas, show=True, printout=False, num_data=300):
    # Keeps track of out in and out error for different lambda values
    mean_sq_err = np.zeros([3, len(lambdas)])
    for i, number in enumerate(lambdas):
        # Standardizing the data
        input_data = stand_n(input_data, num_data)
        # Splitting it into training and testing data
        train, test = split_n(input_data, num_data)
        # features and labels in the training and test data. Features in the training data are the design matrix
        X = train[:, :-1]
        y = train[:, -1]
        test_feats = test[:, :-1]
        test_lab = test[:, -1]

        clf = Ridge(alpha=number)
        clf.fit(X, y)
        weights = clf.coef_

        # Finding the error for both the training and testing data
        in_err = estimate(X, y, weights)
        out_err = estimate(test_feats, test_lab, weights)
        # Recording said errors
        mean_sq_err[0, i] = number
        mean_sq_err[1, i] = in_err
        mean_sq_err[2, i] = out_err
        print("Number is " + str(mean_sq_err[0, i]) + " i is " + str(i))
        print("Input error is " + str(mean_sq_err[1, i]) + " i is " + str(i))
        print("Output error is " + str(mean_sq_err[2, i]) + " i is " + str(i))
        # Printing out the weights, if specified
        # if printout:
        #     power = int((len(input_data[0]) - 1) / 13)
        #     print("\nWeights for values (working with " + str(number) + " data points)")
        #     for j in range(len(info_types) - 1):
        #         print(str(info_types[j]) + " has weight " + str(weights[j]))
        #         for pow in range(2, power + 1):
        #             idx = j + (len(info_types) - 1) * (pow - 1)
        #             print(str(info_types[j]) + "^" + str(pow) + " has weight " + str(weights[idx]))
    if show:
        x_ax_data = np.log10(mean_sq_err[0])
        plt.scatter(x_ax_data, mean_sq_err[1], label="ridge training error")
        plt.scatter(x_ax_data, mean_sq_err[2], label="ridge testing error")
        plt.legend()
        plt.ylabel("Error")
        plt.xlabel("log10(lambda)")
        plt.show()
    return mean_sq_err


def echo(val):
    return val


if __name__ == "__main__":
    # Gathering data
    read = pd.read_table("housing.dat", sep="\s+")
    data = np.array(read)
    np.random.shuffle(data)
    # norm_data = stand(data)
    expanded_data = add_powers(data, 6)
    if 1 != echo(1):
        # One dimensional case
        train_sizes = [25, 50, 75, 100, 150, 200, 300]
        in_out_error(data, train_sizes, show=True, printout=False)

        # Testing higher order polynomials
        powers = np.arange(1, 7)
        # all_data = np.array([int(512 * 4 / 5)])
        in_out_multi(data, train_sizes, powers)

    # Testing ridge regression
    logs = np.linspace(-10, 10, 10)
    lambdas = 10 ** logs
    multi_ridge_reg(expanded_data, lambdas)
