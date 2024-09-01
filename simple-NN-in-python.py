import numpy as np

x = np.array([[200, 17]])
W = np.array([[1, -3, 5], [2, 4, -6]])
b = np.array([[ -1, 1, 2]])

def dense(a_in, w, b): 
    return (1 / (1 + np.exp(-(np.matmul(a_in, w) + b)))).astype(int)

def sequential(x):
    a1 = dense(x , W[0], b)
    a2 = dense(a1, W[2], b)
    f_x = a2
    return f_x