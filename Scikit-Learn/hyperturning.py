import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Movie_regression.csv', header=0)
print(df.head())

#missing values imputation

df['Time_taken'].fillna(value = df['Time_taken'].mean(), inplace=True)

#convert categorical values to numerical

df = pd.get_dummies(df, columns=['3D_available', 'Genre'], drop_first=True)

#X,y
X = df.loc[:, df.columns != 'Collection']
y = df['Collection']

from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

#standardize the datasets, other words is z-score normalization

from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#svm with hyperparameter tuning
from sklearn.model_selection import  GridSearchCV

poly_params = {'C':[0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100, 500, 1000], 'degree':[2,3,4]}

from sklearn.svm import SVC
#poly kernel
svc = SVC(kernel='poly')
svc_grid_lin = GridSearchCV(svc, poly_params , n_jobs=-1, cv=10, verbose=1, scoring='accuracy')

svc_grid_lin.fit(X_train_std, y_train)
print(svc_grid_lin.best_params_)

linsvc_clf = svc_grid_lin.best_estimator_

from sklearn.metrics import accuracy_score

print(accuracy_score(linsvc_clf(y_test, linsvc_clf.predict(X_test_std))))

#radial kernel

radial_params = {'gamma':[0.3,0.4,0.5,0.6], 'C':[0.001, 0.005, 0.01, 0.1, 0.5, 1, 5,]}

svc2 = SVC(kernel='poly')
svc2_grid_lin = GridSearchCV(svc, radial_params, n_jobs=-1, cv=10, verbose=1, scoring='accuracy')

svc2_grid_lin.fit(X_train_std, y_train)
print(svc2_grid_lin.best_params_)

linsvc_clf2 = svc2_grid_lin.best_estimator_

from sklearn.metrics import accuracy_score
print(accuracy_score(linsvc_clf2(y_test, linsvc_clf2.predict(X_test_std))))