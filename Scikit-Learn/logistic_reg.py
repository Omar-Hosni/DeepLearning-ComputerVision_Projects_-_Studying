import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("D:/Projects/Machine Learning & Deep Learning in Python & R/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv")


X = df[['price']]
y = df['Sold']

clf_lrs = LogisticRegression()

clf_lrs.fit(X,y)


#attributes of our model
print(clf_lrs.coef)
print(clf_lrs.intercept_)


#second method to create logistic regression
import statsmodels.api as sn
import statsmodels.discrete.discrete_model as sm

X_cons = sn.add_constant(X)

logit = sm.Logit(y, X_cons).fit()
logit.summary()


#logistic regression with multiple variables

X_multi = df.loc[:, df.columns != 'Sold']
y_multi = df['Sold']

clf_lrs_multi = LogisticRegression()

clf_lrs_multi.fit(X_multi,y_multi)


#attributes of our model
print(clf_lrs.coef)
print(clf_lrs.intercept_)

X_cons_multi = sn.add_constant(X_multi)

logit_multi = sm.Logit(y, X_cons_multi).fit()
logit_multi.summary()

#predicting and confusion matrix
clf_lrs.predict_proba(X)
y_pred = clf_lrs.predict(X)
y_pred_03 = (clf_lrs.predict_proba(X)[:,1] >= 0.3).astype(bool)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y,y_pred))

#calculate performing metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score

#precision = TP/TP+FP
#recall = TP/TP+FN

print(precision_score(y,y_pred))
print(recall_score(y, y_pred))
print(roc_auc_score(y,y_pred))
