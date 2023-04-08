from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist

X_train = mnist.train_images()
y_train = mnist.train_labels()

X_test = mnist.test_images();
y_test = mnist.test_labels();


X_train = X_train.reshape((-1,28*28))
X_test = X_train.reshape((-1,28*28))

X_train = X_train/256
X_test = X_test/256

mlp = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))
mlp.fit(X_train, y_train)

pred = mlp.predict(X_test)
acc = confusion_matrix(y_test, pred)

def accuracy(cm):
    diagonal = cm.trace()
    elem = cm.sum()
    return diagonal/elem

print('acc ', print(accuracy(acc)))


