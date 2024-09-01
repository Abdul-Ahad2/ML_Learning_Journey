import matplotlib.pyplot as plt
import  numpy as np


# Normal Example
# x_init = np.random.random(50) * 100
# y_init = np.random.random(50) * 100

# plt.scatter(x_init, y_init, c="r", marker='x', alpha=1)
# plt.show()


years = [ x + 2006 for x in range(16)]
weight = [round(np.random.uniform(80, 90), 2) for _ in range(16)]



plt.plot(years, weight, c='b')
plt.title("Increase in weight")
plt.xlabel("years")
plt.ylabel("weights")


plt.scatter(years, weight, c="r", marker='x', alpha=1)
plt.show()

