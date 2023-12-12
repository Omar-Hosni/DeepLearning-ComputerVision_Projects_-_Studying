import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

import keras_tuner as kt

mnist = mnist.load_data()
X_train, y_train, X_test, y_test = train_test_split(mnist, test_size=0.2, random_state=42, shuffle=True)

def model_builder(hp):
    model = keras.Sequential()

    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)

    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='output_model',
                     project_name='intro_to_kt',
                     )

stop_early_fn = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early_fn])
