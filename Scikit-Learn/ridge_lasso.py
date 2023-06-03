from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv("D:/Projects/Machine Learning & Deep Learning in Python & R/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv")

X_multi = df.drop("price", axis=1)
y_multi = df['price']

X_train, y_train, X_test, y_test = train_test_split(X_multi,y_multi, test_size=0.2, random_state=1234)
lm = LinearRegression()
lm.fit(X_train, y_train)

lm.predict(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

from sklearn.linear_model import Ridge

lm_r = Ridge(alpha=0.5)
lm_r.fit(X_train_s, y_train)
r2_score(y_test, lm_r.predict(X_test_s))

from sklearn.model_selection import validation_curve

param_range = np.logspace(-2, 8, 100)

train_score, test_score = validation_curve(Ridge(), X_train_s, y_train, "alpha", param_range, scoring='r2')

train_mean = np.mean(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)

np.where(test_mean == max(test_mean))

lm_r_best = Ridge(alpha=param_range[33])

lm_r_best.fit(X_train, y_train)

print(r2_score(y_test, lm_r_best.predict(X_test_s)))
print(r2_score(y_train, lm_r_best.prdict(X_train_s)))

from sklearn.linear_model import Lasso

lm_l = Lasso(alpha=0.4)









