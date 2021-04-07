import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Used to plot individual features to see what impacts MEDV most directly
def make_plot():
    med_value = np.array(data["MEDV"])
    plt.ylabel("Median Value ($1000s)")
    column = "LSTAT"
    plt.xlabel(" % lower status of the population")
    x = np.array(data[column])
    plt.scatter(x, med_value)
    # plt.savefig("housing/" + column)
    # plt.show()


# standardizes the first n elements by subtracting the mean and dividing by the standard
# deviation. Then standardizes the remaining (test) data using the same mean and standard deviation
def stand_n(array, n):
    mean = np.mean(array[:n])
    std_dev = np.std(array[:n])
    training = (array[:n] - mean) / std_dev
    test = (array[n:] - mean) / std_dev
    return training, test


# Finding the best of the labels using the given weights, and finding the mean squared error
def estimate(design, labels, weights):
    est = np.matmul(design, weights)
    dif = labels - est
    dif_sq = np.dot(dif, dif)
    # mean squared error
    err = np.sum(dif_sq) / len(labels)
    return err


# appends to the array so that we can train on higher order terms (ignores interaction terms)
def add_powers(array, power):
    assert power > 0
    array = np.transpose(array)
    result = array
    ones = np.ones([len(array), len(array[0])])
    for i in range(2, power + 1):
        result = np.vstack((result, array ** (ones * i)))
    return np.transpose(result)


def test_add_powers():
    test = np.ones([2, 3]) * 2
    test[0, 2] = -1
    # print(test)
    print("Add powers gave me " + str(add_powers(test, 1)))


# Finds the training and generalization error for some data, at each split
def in_out_error(data, splits):
    # Keeps track of out in and out error for different sizes of training data
    mean_sq_err = np.zeros([3, len(splits)])
    for i, number in enumerate(splits):
        train, test = stand_n(data, number)
        # features and labels in the training and test data. Features in the training data are the design matrix
        design = train[:, :-1]
        train_lab = train[:, -1]
        test_feats = test[:, :-1]
        test_lab = test[:, -1]
        # the transpose and inverse
        design_t = np.matrix.transpose(design)
        square = np.matmul(design_t, design)
        print(square)
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
    plt.scatter(mean_sq_err[0], np.log(mean_sq_err[1]), label="log-training error")
    plt.scatter(mean_sq_err[0], np.log(mean_sq_err[2]), label="log-testing error")
    plt.legend()
    plt.ylabel("ln(Error)")
    plt.xlabel("Quantity of training data")
    plt.show()


if __name__ == "__main__":
    # Gathering data
    read = pd.read_table("housing.dat", sep="\s+")
    data = np.array(read)
    # TODO: shuffle data once you're done debugging
    # np.random.shuffle(data)

    # One dimensional case
    splits = [25, 50, 75, 100, 150, 200, 300]
    # splits = [25]
    # in_out_error(data, splits)

    # Testing higher order polynomials
    mod_data = add_powers(data, 3)
    in_out_error(mod_data, splits)
