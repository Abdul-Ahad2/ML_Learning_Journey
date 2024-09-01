import numpy as np


x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16]).reshape(-1, 4 )
y_train = np.array([5, 9, 13, 17])


w_init = np.zeros(4)
b_init = 0


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = np.zeros_like(w)
    dj_db = 0
    
    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db


def compute_gradient_descent(x, y, w, b, itr, alpha):
    for i in range(itr):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w,b    


w_final, b_final = compute_gradient_descent(x_train, y_train, w_init, b_init, 10000, 0.001)



prediction = np.dot(w_final, [100,101, 102, 103]) + b_final
print(prediction)
