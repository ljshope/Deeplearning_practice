# -*- coding: utf-8 -*-

#import random
import tensorflow as tf
import numpy as np
#random.seed(123) 
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=8, input_dim=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.1),  metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=100)

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])