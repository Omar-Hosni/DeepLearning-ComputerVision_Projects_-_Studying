import os


import numpy as np

import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

X = []
y = []

for j in range(9):
    files = os.listdir("C://Users//Custom PC//Desktop//ML//Scikit-Learn/hand_digits/{}".format(j))
    len_f = len(files)

    for i in range(len_f):
        img = Image.open('hand_digits/{}/{}'.format(j, files[i]))
        data = np.array(list(img.getdata))/256
        x = slice(1958, 8042)
        data = data[x].tolist()
        print(data)

clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(400,2400), random_state=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15)
clf.fit(X_train, y_train)

filename = "model.sav"
joblib.dump(clf, filename)
