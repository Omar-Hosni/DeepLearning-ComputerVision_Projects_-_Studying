import keras.models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, concatenate
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

keras.backend.clear_session()
np.random.seed(42)
X = load_iris.data[:, -1]
y = load_iris.target

X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)


model = keras.models.Sequential([
    keras.layers.Dense(30,activation='relu', input_shape=[8]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=1e-3))

checkpoint_cb = keras.callbacks.ModelCheckpoint('my_func_model.h5')

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint_cb])


mse_test = model.evaluate(X_test, y_test)
