import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Movie_regression.csv", header=0)

#missing value imputation
df['Time_taken'].fillna(value = df['Time_taken'].mean(), inplace=True)

#dummy variable creation
df = pd.get_dummies(df, columns=['3D_available','Genre'], drop_first=True)

#X-y split
X = df.loc[:, df.columns != 'Collection']
y = df['Collection']

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

reg_tree = tree.DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_train, y_train)
y_pred = reg_tree.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

#plotting decision tree

from IPython.display import Image
import pydotplus

dot_data = tree.export_graphviz(reg_tree, out_fiile=None)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())




