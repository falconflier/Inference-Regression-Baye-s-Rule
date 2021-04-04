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


# Things that have worked! (They give acceptance rates of ~23.4%
# path = sampler(100, 10000, 3)
