#linear discriminant analysis
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import pandas as pd

df = pd.read_csv("D:/Projects/Machine Learning & Deep Learning in Python & R/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv")


X = df[['price']]
y = df['Sold']

clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X,y)
y_pred_lda = clf_lda.predict(X)
print(confusion_matrix(y,y_pred_lda))