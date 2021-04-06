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


if __name__ == "__main__":
    print(find_ratio())
