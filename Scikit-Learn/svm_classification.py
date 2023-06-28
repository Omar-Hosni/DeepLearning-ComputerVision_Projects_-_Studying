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

#svm
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=0.01) #C = regularization parameter
svc.fit(X_test_std, y_train)

y_train_pred = svc.predict(X_train_std)
y_test_pred = svc.predict(X_test_std)

from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, y_test_pred))
print(r2_score(y_train, y_train_pred))
print(r2_score(y_test, y_test_pred))



