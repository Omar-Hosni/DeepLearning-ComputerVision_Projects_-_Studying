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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

ada_clf = AdaBoostClassifier(learning_rate=0.02, n_estimators=5000)
ada_clf.fit(X_train, y_train)
print(accuracy_score(y_train, ada_clf.predict(X_train)))

rf_clf = RandomForestClassifier(n_estimators=250, random_state=42)
ada_clf2 = AdaBoostClassifier(rf_clf, learning_rate=0.02, n_estimators=500)
ada_clf2.fit(X_train, y_train)
print(accuracy_score(y_test, ada_clf2.predict(X_test)))
