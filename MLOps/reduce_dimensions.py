import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import reuters
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


num_words = 1000

reuters_data = reuters.load_data(num_words=num_words)
X_train, y_train, X_test, y_test = train_test_split(reuters_data,
                                                    train_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)

n_labels = np.unique(y_train).shape[0]

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 46)
y_test = np_utils.to_categorical(y_test, 46)

from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Embedding(num_words, 1000, input_length=20),
    layers.Flatten(),
    layers.Dense(256),
    layers.Dropout(0.25),
    layers.Activation('relu'),
    layers.Dense(46),
    layers.Activation('softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model = model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  batch_size=128, epochs=20, verbose=0)