import numpy as np

# Data
x_train = np.array([3,3,9,4,4,16,5,6,25,6,6,36,7,7,49,8,7,64,9,9,81,10,10,100,12,10,144,2,0,4,1,2,1]).reshape(-1,3)
y_train = [1,1,0,1,1,0,1,1,0,0,0]

# Normalization parameters
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train_norm = (x_train - mean) / std

print("Normalized Training Data:\n", x_train_norm)

# Initialize parameters
w_init = np.zeros(3)
b_init = 0

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = np.zeros_like(w)
    dj_db = 0
    
    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def compute_gradient_descent(x, y, w, b, itr, alpha):
    for i in range(itr):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b 

# Train the model
w_final, b_final = compute_gradient_descent(x_train_norm, y_train, w_init, b_init, 10000, 0.01)

# Testing with a new data point
let_x = np.array([5,6,25])
x_test = (let_x - mean) / std  # Normalize using training mean and std
print("Normalized Test Data:\n", x_test)

# Prediction
prediction = np.dot(w_final, x_test) + b_final
print("Prediction:\n", prediction)
