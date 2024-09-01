import numpy as np

x = np.array([[200, 17]])
W = np.array([[1, -3], [2, 4]])
b = np.array([[-1, 1]])

def dense(a_in, w, b): 
    return (1 / (1 + np.exp(-(np.matmul(a_in, w) + b)))).astype(int)

def sequential(x):
    a1 = dense(x, W, b)
    a2 = dense(a1, W, b)
    f_x = a2
    return f_x

result = sequential(x)
print(result)
