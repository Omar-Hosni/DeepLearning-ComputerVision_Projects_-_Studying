'''
dataset: UCI car evaluation
download data folder from UCI, car.data
'''

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data = pd.read_csv('car.data')
X = data[['buying', 'maint','safety']].values
y = data[['class']]

print(X, y)

#convert to numbers, because ML algorithms can not train on characters

Le = LabelEncoder()

for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])


#y
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)


#create model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)
print('predictions: ', predictions)
print('accuracy: ',accuracy)

a = 100
print('actual value: ', y[a])
print('predicted value: ', knn.predict(X)[a])

#another way
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_s = scaler.transform(X_train)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_s = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

clf_knn = KNeighborsClassifier(n_neighbors=1)
clf_knn.fit(X_train_s, y_train)

print(confusion_matrix(y_test, clf_knn.predict(X_test_s)))
print(accuracy_score(y, clf_knn.predict(X_test_s)))

clf_knn_3 = KNeighborsClassifier(n_neighbors=3)
clf_knn_3.fit(X_train_s, y_train)
print(accuracy_score(y_test, clf_knn_3.predict(X_test_s)))


#multiple values of K
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors' :[1,2,3,4,5,6,7,8,9,10]}

grid_search_cv = GridSearchCV(KNeighborsClassifier(), params)
grid_search_cv.fit(X_train_s, y_train)
optimised_KNN = grid_search_cv.best_estimator_
y_test_pred = optimised_KNN.predict(X_test_s)
print(confusion_matrix(X_test, y_test_pred))
print(accuracy_score(y_test, y_test_pred))






