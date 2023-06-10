import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
#import pydotplus
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


df = pd.read_csv("Movie_classification.csv", header=0)

df['Time_taken'].fillna(value=df['Time_taken'].mean(), inplace= True)
df = pd.get_dummies(df, columns = ['3D_available', 'Genre'], drop_first=True)

#X-y split
X = df.loc[:, df.columns != 'Start_Tech_Oscar']
y = df['Start_Tech_Oscar']

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


rf_clf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)

#grid search-hyperparameter tuning

from sklearn.model_selection import GridSearchCV

rf_clf = RandomForestClassifier(n_estimators=250, random_state=42)
params_grid = {"max_features":[4,5,6,7,8,9,10], "min_samples_split":[2,3,10]}

grid_search = GridSearchCV(rf_clf, params_grid, n_jobs=-1, cv=5, scoring='accuracy')
print(grid_search.best_params_)

cvrf_clf = grid_search.best_estimator_
print(accuracy_score(y_test, cvrf_clf.predict(X_test)))
print(confusion_matrix(y_test, cvrf_clf.predict(X_test)))