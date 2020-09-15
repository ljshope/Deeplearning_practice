# -*- coding: utf-8 -*-
 
import numpy as np
import random
import tensorflow as tf

random.seed(123) 
learning_rate = 0.001
batch_size = 128
training_epochs = 12
nb_classes = 10
drop_rate = 0.3

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test/255
x_train = x_train/255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(3,3), input_shape = (28,28,1), activation='relu'))
tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
tf.model.add(tf.keras.layers.Dropout(drop_rate))

tf.model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(3,3), input_shape = (28,28,1), activation='relu'))
tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
tf.model.add(tf.keras.layers.Dropout(drop_rate))

tf.model.add(tf.keras.layers.Flatten())
tf.model.add(tf.keras.layers.Dense(units=nb_classes, kernel_initializer='glorot_normal', activation='softmax'))
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
tf.model.summary()

tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# predict 10 random hand-writing data
y_predicted = tf.model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

# evaluate test set
evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
