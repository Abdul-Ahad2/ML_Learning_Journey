from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x_init = np.array([x + 2000 for x in range(11)]).reshape(-1,1)
y_init = np.array([56, 60, 67, 68, 61, 70, 78, 80, 74, 85, 90])


model = LinearRegression()

model.fit(x_init, y_init)

yhat = model.predict(x_init)

plt.scatter(x_init, y_init, color='blue', label='Data Points')
plt.plot(x_init, yhat, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()