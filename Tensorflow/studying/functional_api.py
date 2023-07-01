import keras.models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, concatenate
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X = load_iris.data[:, -1]
y = load_iris.target

X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)


input_ = Input(shape=X_train.shape[1:])
hidden1 = Dense(30, activation='relu')(input_)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = concatenate([input_, hidden2])
output = Dense(1)(concat)
model = tf.keras.models.Model(inputs=[input_], outputs=[output])

model.summary()
model.compile(loss='mean_squared_errpr', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), metrics=['mae'])
model_history = model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

model.save('my_func_model.h5')
keras.backend.clear()

model = keras.models.load_model('my_func_model.h5')

