# Without Vectorization
import numpy as np
import matplotlib.pyplot as plt

x = np.array([2,4,6,8])
y = np.array([4,6,8,10])

def compute_cost( x, y, w, b):
  m = len(x)
  total_sum = np.sum((w * x + b - y) ** 2)
  return ((1/(2 * m)) * total_sum)

def cal_gradient_descent(x, y, w, b, itr, alpha):
  m = len(x)
  for i in range(itr):
    if i%1000 == 0:
      print(f"{i/1000+1:.0f})cost: {compute_cost( x, y, w, b):.7e}, w: {w:.2e}, b: {b:.2e}")
    w -= (alpha / m) * sum(((w * x + b) - y) * x) 
    b -= (alpha / m) * sum((w * x + b) - y) 
    
  return w, b

w,b = cal_gradient_descent(x, y, 0, 0, 10000, 0.01)
print(f"w: {w}\tb: {b}")


x_fit = np.linspace(min(x), max(x), 2)
y_fit = w * x_fit + b

#Check you predictive value below: 
# predict_Y_when_X = 6
# print(f"x: {predict_Y_when_X}\t{(w * predict_Y_when_X + b):.2f}")

plt.scatter(x, y, label='Data points', c="r")
plt.plot(x_fit, y_fit, color='blue', label='Fitted line')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show() 

    


