import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
#import pydotplus
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("Movie_classification.csv", header=0)

df['Time_taken'].fillna(value=df['Time_taken'].mean(), inplace= True)
df = pd.get_dummies(df, columns = ['3D_available', 'Genre'], drop_first=True)

#X-y split
X = df.loc[:, df.columns != 'Start_Tech_Oscar']
y = df['Start_Tech_Oscar']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)



xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3, n_jobs=-1)
xgb_clf.fit(X_train, y_train)
print(accuracy_score(y_train, xgb_clf.fit(X_train)))

param_test1={
    'max_depth':range(3,10,2),
    'gamma':[0.1, 0.2, 0.3],
    'subsample':[0.8, 0.9],
    'colsample_bytree':[0.8, 0.9],
    'reg_alpha':[1e-2, 0.1, 1]
}
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs=-1, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

cross_validated_XGB = grid_search.best_estimator_
print(grid_search.best_params_)
