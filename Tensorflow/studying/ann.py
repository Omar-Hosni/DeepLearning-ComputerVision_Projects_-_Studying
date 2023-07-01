import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', ' Pullover', 'Dress', 'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the input images
model.add(Dense(128, activation='relu'))  # Hidden layer with 128 neurons and ReLU activation
model.add(Dense(10, activation='softmax'))  # Output layer with 10 neurons (10 classes) and softmax activation

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))