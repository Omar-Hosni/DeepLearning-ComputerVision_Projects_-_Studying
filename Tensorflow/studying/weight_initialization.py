import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X = load_iris.data[:, -1]
y = load_iris.target

X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()


        self.fc1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu,
                                         kernel_initializer='glorot_uniform',
                                         bias_initializer='zeros')

        self.fc1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu,
                                         kernel_initializer='glorot_uniform',
                                         bias_initializer='zeros')

        self.fc1 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu,
                                         kernel_initializer='glorot_uniform',
                                         bias_initializer='zeros')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = NeuralNetwork()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

#perform forward and backward propagation
with tf.GradientTape() as tape:
    logits = model(X_train)
    loss_value = loss_fn(y_train, logits)

gradients = tape.gradient(loss_value, model.train_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
