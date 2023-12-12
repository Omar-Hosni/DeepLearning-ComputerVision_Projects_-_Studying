# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 07:44:01 2020

Task: Decision tree analysis of Iris data

Python tools    
Libraries: matplotlib, sklearn
Modules: pyplot, tree
Classes: DecisionTreeClassifier
Functions: plot_tree

@author: Márton Ispány
"""

from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.datasets import load_iris; # importing iris loader
from sklearn.tree import DecisionTreeClassifier, plot_tree;    # importing decision tree tools

# Loading the dataset
iris = load_iris();

# Initialize our decision tree object
crit = 'entropy';
depth =3;
# Instance of decision tree class
class_tree = DecisionTreeClassifier(criterion=crit,max_depth=depth);

# Fitting decision tree (tree induction + pruning)
class_tree.fit(iris.data, iris.target);
score_entropy = class_tree.score(iris.data, iris.target); # Goodness of tree

# Visualizing decision tree
fig = plt.figure(1,figsize = (12,6),dpi=100);
plot_tree(class_tree, feature_names = iris.feature_names, 
               class_names = iris.target_names,
               filled = True, fontsize = 8);
fig.savefig('iris_tree_entropy.png'); # Writing to local repository as C:\\Users\user_name 

# Initialize our decision tree object
crit = 'gini';
depth =3;
# Instance of decision tree class
class_tree = DecisionTreeClassifier(criterion=crit,max_depth=depth);

# Fitting decision tree (tree induction + pruning)
class_tree.fit(iris.data, iris.target);
score_gini = class_tree.score(iris.data, iris.target); # Goodness of tree

# Visualizing decision tree
fig = plt.figure(1,figsize = (12,6),dpi=100);
plot_tree(class_tree, feature_names = iris.feature_names, 
               class_names = iris.target_names,
               filled = True, fontsize = 8);
fig.savefig('iris_tree_gini.png'); # Writing to local repository as C:\\Users\user_name 

