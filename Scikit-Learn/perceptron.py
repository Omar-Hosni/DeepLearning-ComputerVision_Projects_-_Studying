import numpy as np
import pandas as  pd
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, (2,3)] #petal length, petal width. the 3rd and 4th column
y = (iris.target == 0)

from sklearn.linear_model import Perceptron

p = Perceptron(random_state=42)
p.fit(X,y)
y_pred=p.predict(y)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y))




