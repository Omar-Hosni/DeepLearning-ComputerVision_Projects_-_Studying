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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)

print(accuracy_score(y_train, gbc_clf.predict(X_train)))


gbc_clf2 = GradientBoostingClassifier(learning_rate=0.02, n_estimators=1000, max_depth=1)
gbc_clf2.fit(X_train, y_train)

print(accuracy_score(y_test, gbc_clf2(X_test)))

