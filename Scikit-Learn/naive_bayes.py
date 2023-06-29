import numpy as np
import pandas as pd
import requests

import sklearn
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
response = requests.get(url)
raw_data = response.content.decode('utf-8')

dataset = np.loadtxt(raw_data, delimeter=',')

X = dataset[:, 48]
y = dataset[:, -1]

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

BernNB = BernoulliNB(binarize=True)
BernNB.fit(X_train, y_train)
print(accuracy_score(y_test, BernNB.predict(y_test)))


MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
print(accuracy_score(y_test, MultiNB.predict(y_test)))

GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
print(accuracy_score(y_test, GausNB.predict(y_test)))





