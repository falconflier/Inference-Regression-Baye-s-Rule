import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_table("housing.dat", sep="\s+")

med_value = np.array(data["MEDV"])
plt.ylabel("Median Value ($1000s)")

column = "LSTAT"
plt.xlabel(" % lower status of the population")

x = np.array(data[column])
plt.scatter(x, med_value)

# plt.savefig("housing/" + column)
# plt.show()
