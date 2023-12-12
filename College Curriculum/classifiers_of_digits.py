# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:37:40 2020

Task: Fitting classifiers for Digits dataset
Classifiers: logistic regression, naive Bayes, nearest neighbor, neural network

Python tools    
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, linear_model, naive_bayes, model_selection, metrics
Classes: LogisticRegression, GaussianNB
Functions: urlopen, train_test_split

@author: Márton Ispány
"""

from sklearn import datasets as ds; # importing scikit-learn datasets
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier;    # importing nearest neighbor classifier
from sklearn.neural_network import MLPClassifier; # importing neural network classifier

# loading dataset
digits = ds.load_digits();
n = digits.data.shape[0];  # number of records
p = digits.data.shape[1];  # number of attributes

# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, 
            test_size=0.3, shuffle = True, random_state=2020);

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);  #  goodness of fit
score_test_logreg = logreg_classifier.score(X_test,y_test);  #  goodness of fit
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities

# Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
score_train_naive_bayes = naive_bayes_classifier.score(X_train,y_train);
score_test_naive_bayes = naive_bayes_classifier.score(X_test,y_test);  #  goodness of fit
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities

# Fitting nearest neighbor classifier
K = 5;  # number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=K);
knn_classifier.fit(X_train,y_train);
score_train_knn = knn_classifier.score(X_train,y_train);  #  goodness of fit
score_test_knn = knn_classifier.score(X_test,y_test);  #  goodness of fit
ypred_knn = knn_classifier.predict(X_test);   # spam prediction
yprobab_knn = knn_classifier.predict_proba(X_test);  #  prediction probabilities

# Fitting neural network classifier
neural_classifier = MLPClassifier(hidden_layer_sizes=(16),activation='logistic',solver='lbfgs',max_iter=5000);  #  number of hidden neurons: 16
neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);
score_test_neural = neural_classifier.score(X_test,y_test);  #  goodness of fit
ypred_neural = neural_classifier.predict(X_test);   # spam prediction
yprobab_neural = neural_classifier.predict_proba(X_test);  #  prediction probabilities

#  The best model based on train score is Nearest neighbor with 99.8%
#  The best model based on test score is Logistic Regression with 97.4%
