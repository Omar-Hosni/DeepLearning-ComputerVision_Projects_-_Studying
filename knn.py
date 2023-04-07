'''
dataset: UCI car evaluation
download data folder from UCI, car.data
'''

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
