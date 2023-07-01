import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Create an RNN model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32))
model.add(LSTM(units=64))
model.add(Dense(units=10, activation='softmax'))


#OOP

class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=32)
        self.lstm = tf.keras.layers.LSTM(units=64)
        self.fc = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.fc(x)
        return x

# Create an instance of the RNN model
model = RNN()