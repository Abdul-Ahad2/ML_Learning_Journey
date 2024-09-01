import numpy as np   
import matplotlib.pyplot as plt

x_train = np.array([3, 3, 2, 2, 1, 5, 7, 7, 7, 8, 6, 9, 9, 8, 9, 14]).reshape(-1, 4)
y_train = np.array([1, 2, 3, 4])
w_init = np.zeros(4)
b_init = 0

def compute_cost(x, y, w, b):
    cost = 0.0
    m = x_train.shape[0]
    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        cost  += (f_wb - y[i]) ** 2
    return (1 / (2 * m)) * cost

def compute_gradient(x, y, w, b):
    sum_w = 0.0
    sum_b = 0.0
    m = x_train.shape[0]
    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        sum_w += (f_wb - y[i]) * x[i]
        sum_b += f_wb - y[i]
    return (1/m) * sum_w, (1/m) * sum_b

def compute_gradient_descent(x, y, w, b, itr, alpha):
    for i in range(itr):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b

w_final, b_final = compute_gradient_descent(x_train, y_train, w_init, b_init, 10000, 0.01)



print(f"{np.dot(w_final, [55, 50, 1, 94]) + b_final:.2f}")
# print(F"{x_train}\n{y_train.reshape(-1,1)}")





