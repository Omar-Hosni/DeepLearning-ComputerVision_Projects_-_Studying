# -*- coding: utf-8 -*-
"""
Created on Sun March 29 23:02:03 2020

Task: Decision tree analysis of Spambase data reading from URL
Original data source: https://archive.ics.uci.edu/ml/datasets/spambase

Python tools    
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, model_selection, tree
Classes: DecisionTreeClassifier
Functions: urlopen, train_test_split, plot_tree

@author: Márton Ispány
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.model_selection import train_test_split; # importing model selection tools
from sklearn.tree import DecisionTreeClassifier, plot_tree;    # importing decision tree tools

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
X = data[:,0:56];
y = data[:,57];
del data;
input_names = attribute_names[0:56];
target_names = ['not spam','spam'];


# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                shuffle = True, random_state=2020);

# Initialize the decision tree object
crit = 'gini';
depth =4;
# Instance of decision tree class
class_tree = DecisionTreeClassifier(criterion=crit,max_depth=depth);

# Fitting decision tree on training dataset
class_tree.fit(X_train, y_train);
score_train = class_tree.score(X_train, y_train); # Goodness of tree on training dataset
score_test = class_tree.score(X_test, y_test); # Goodness of tree on test dataset

# Predicting spam for test data
y_pred_gini = class_tree.predict(X_test);

# Visualizing decision tree
fig = plt.figure(1,figsize = (16,10),dpi=100);
plot_tree(class_tree, feature_names = input_names, 
               class_names = target_names,
               filled = True, fontsize = 6);
# Writing to local repository as C:\\Users\user_name
fig.savefig('spambase_tree_gini.png');  

# Initialize the decision tree object
crit = 'entropy';
depth =4;
# Instance of decision tree class
class_tree = DecisionTreeClassifier(criterion=crit,max_depth=depth);

# Fitting decision tree (tree induction + pruning)
class_tree.fit(X_train, y_train);
score_entropy = class_tree.score(X_train, y_train); # Goodness of tree on training dataset
score_test = class_tree.score(X_test, y_test); # Goodness of tree on test dataset

# Predicting spam for test data
y_pred_entropy = class_tree.predict(X_test);

# Visualizing decision tree
fig = plt.figure(2,figsize = (16,10),dpi=100);
plot_tree(class_tree, feature_names = input_names, 
               class_names = target_names,
               filled = True, fontsize = 6);
# Writing to local repository as C:\\Users\user_name 
fig.savefig('spambase_tree_entropy.png'); 


