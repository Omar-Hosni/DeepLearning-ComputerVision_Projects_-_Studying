from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
california = datasets.fetch_california_housing()

#features / labels
X = california.data
y = california.target

print(X.shape)
print(y.shape)

l_reg = LinearRegression()

plt.scatter(X.T[4], y)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

l_reg.fit(X_train, y_train)

predictions = l_reg.predict(X_test)
#acc = accuracy_score(y_test, predictions)
print('r^2 value: ', l_reg.score(X,y))
print('coedd: ', l_reg.coef_) #slope

