import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Movir_classification.csv", header=0)

df['Time_taken'].fillna(value=df['Time_taken'].mean(), inplace= True)
df = pd.get_dummies(df, columns = ['3D_available', 'Genre'], drop_first=True)

#X-y split
X = df.loc[:, df.columns != 'Start_Tech_Oscar']
y = df['Start_Tech_Oscar']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clftree = DecisionTreeClassifier(max_depth=3)

clftree.fit(X_train, y_train)
y_train_pred = clftree.predict(X_train)
y_test_pred = clftree.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_train, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(accuracy_score(y_test, y_test_pred))

dot_data = tree.export_graphviz(clftree, out_file=None, feature_names=X_train.columns, filled=True)
from IPython.display import Image
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

#Controlling tree grow

clftree2 = DecisionTreeClassifier(min_samples_leaf = 20, max_depth=4)
clftree2.fit(X_train, y_train)
dot_data = tree.export_graphviz(clftree2, out_file=None, feature_names=X_train.columns, filled=True)
graph2 = pydotplus.graph_from_dot_data(dot_data)
Image(graph2.create_png())




