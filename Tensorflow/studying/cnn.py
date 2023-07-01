import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a CNN model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

#OOP
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Create an instance of the CNN model
model = CNN()