print('Importing packages... ', end='')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
print('Done')

print('Reading data... ', end='')
data = pd.read_pickle('./data_full.pkl').astype('float32')
X = data.drop('exercise', axis=1)
y = data.loc[:, 'exercise']
print('Done')

print('Splitting data... ', end='')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Done')

# 132 input nodes * 4 = 528 hidden nodes

print('Compiling model... ', end='')
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(22)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print('Done')

model.fit(X_train, y_train, epochs=20)

print('\n')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_accuracy)

print('Saving model... ', end='')
model.save('model')
print('Done')