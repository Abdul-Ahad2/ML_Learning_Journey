import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers,losses


X = np.array([[1,1,0],[1,0,1], [1,1,1], [0,0,1], [1,0,0], [0,1,1], [1,1,1], [1,1,1]])
Y = np.array([[0,0,1,0,0,0,1,1]]).T 

model = Sequential([
    layers.Dense(units=25, activation='sigmoid'),
    layers.Dense(units=15, activation='sigmoid'),
    layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss=losses.BinaryCrossentropy())

model.fit(X,Y, epochs=1000)

model.summary()

prediction = (model.predict(np.array([[1,1,1]])) > 0.5).astype(int)

print(prediction)
