import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
#import pydotplus


df = pd.read_csv("Movie_classification.csv", header=0)

df['Time_taken'].fillna(value=df['Time_taken'].mean(), inplace= True)
df = pd.get_dummies(df, columns = ['3D_available', 'Genre'], drop_first=True)

#X-y split
X = df.loc[:, df.columns != 'Start_Tech_Oscar']
y = df['Start_Tech_Oscar']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


from sklearn import tree

clf_tree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(base_estimator=clf_tree, n_estimators=1000, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test, bag_clf.predict(X_test)))
print(accuracy_score(y_test, bag_clf.predict(X_test)))