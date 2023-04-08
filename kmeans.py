from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd


bc = load_breast_cancer()

X = scale(bc.data)
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y)

km = KMeans(n_clusters=2, random_state=0)
km.fit(X_train, y_train)

predictions = km.predict(X_test)
labels = km.labels_

print('acc ', accuracy_score(y_test, predictions))






