# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 09:36:05 2020

Task: Fitting classifiers for Spambase dataset
Classifiers: logistic regression, naive Bayes, nearest neighbor, neural network
Original data source: https://archive.ics.uci.edu/ml/datasets/spambase

Python tools    
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, linear_model, naive_bayes, neighbors, neural_network, model_selection
Classes: LogisticRegression, GaussianNB
Functions: urlopen, train_test_split

@author: Márton Ispány
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier;    # importing nearest neighbor classifier
from sklearn.neural_network import MLPClassifier; # importing neural network classifier

# Reading the dataset
url = 'https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spamdata.csv';
raw_data = urlopen(url);
data = np.loadtxt(raw_data, skiprows=1, delimiter=";");  # reading numerical data from csv file
del raw_data;

# Reading attribute names 
url_names = 'https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spambase.names.txt	';
raw_names = urlopen(url_names);
attribute_names = [];   #  list for names
for line in raw_names:
    name = line.decode('utf-8');  # transforming bytes to string
    name = name[0:name.index(':')]; # extracting attribute name from string
    attribute_names.append(name);  # append the name to a list
del raw_names;

# Defining input and target variables
X = data[:,0:57];
y = data[:,57];
del data;
input_names = attribute_names[0:57];
target_names = ['not spam','spam'];


# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                shuffle = True, random_state=2020);

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);
score_test_logreg = logreg_classifier.score(X_test,y_test);  #  goodness of fit
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities

# Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
score_train_naive_bayes = naive_bayes_classifier.score(X_train,y_train);  #  goodness of fit
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
neural_classifier = MLPClassifier(hidden_layer_sizes=(3),activation='logistic',max_iter=500);  #  number of hidden neurons: 5
neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);  #  goodness of fit
score_test_neural = neural_classifier.score(X_test,y_test);  #  goodness of fit
ypred_neural = neural_classifier.predict(X_test);   # spam prediction
yprobab_neural = neural_classifier.predict_proba(X_test);  #  prediction probabilities

#  The best model based on test score is MLP (Multilayer perceptron)
#  with 93.1%

# Visualization of spam prediction and probabilities using the best model (MLP)
# Color denotes the class, size denotes the probability
# Default axis
x_axis = 5;  # x axis attribute (0..56)
y_axis = 22;  # y axis attribute (0..56)
# Enter axis from consol
user_input = input('X axis [0..56, default:5]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=56 :
    x_axis = np.int8(user_input);
user_input = input('Y axis [0..56, default:22]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=56 :
    y_axis = np.int8(user_input);    
colors = ['blue','red'];  #  colors of spam and not spam
fig = plt.figure(1);
plt.title('Scatterplot for training digits dataset');
plt.xlabel(input_names[x_axis]);
plt.ylabel(input_names[y_axis]);
plt.scatter(X_test[:,x_axis],X_test[:,y_axis],s=100*yprobab_neural[0,:],
            c=ypred_neural,cmap=col.ListedColormap(colors));
plt.show();
