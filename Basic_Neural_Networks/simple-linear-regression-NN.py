from matplotlib import axes
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
  

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])


model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())


model.fit(X_train_scaled, y_train, epochs=1000)


yhat_train = model.predict(X_train_scaled)
yhat_test = model.predict(X_test_scaled)  


fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(10, 30))

for i in range(X_train.shape[1]):
    axes[i].scatter(X_test[:, i], y_test, c='b', label='Actual', alpha=0.5)
    axes[i].scatter(X_test[:, i], yhat_test, c='red', label='Predicted', alpha=0.5)
    axes[i].set_title(f"Feature {i+1}")
    axes[i].set_xlabel(f"Feature {i+1}")
    axes[i].set_ylabel("Target")
    axes[i].legend()
    
plt.tight_layout()
plt.show()
