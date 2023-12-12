# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23:34:18 2020

Task: Pricipal Component Analysis (PCA) of Iris data  
Results: 2D plots 

Python tools    
Libraries: numpy, matplotlib, seaborn, sklearn
Modules: pyplot, colors, datasets, feature_selection, decomposition
Classes: SelectKBest, PCA
Functions:

@author: Márton Ispány
"""

import numpy as np;  # importing numerical library
import matplotlib.pyplot as plt;  # importing MATLAB-like plotting framework
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib
import seaborn as sns;  # importing the Seaborn library
from sklearn.datasets import load_iris; # importing Iris dataset
from sklearn.feature_selection import SelectKBest; # importing feature selection 
from sklearn.decomposition import PCA;  # importing PCA
 
# loading dataset
iris = load_iris();
n = iris.data.shape[0]; # number of records
p = iris.data.shape[1]; # number of attributes
k = iris.target_names.shape[0]; # number of target classes

# Printing the basic parameters
print(f'Number of records:{n}');
print(f'Number of attributes:{p}');
print(f'Number of target classes:{k}');

# Scatterplot for two input attributes
# Default axis
x_axis = 0;  # x axis attribute (0,1,2,3)
y_axis = 1;  # y axis attribute (0,1,2,3)
# Enter axis from consol
user_input = input('X axis [0..3, default:0]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    x_axis = np.int8(user_input);
user_input = input('Y axis [0..3, default:1]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    y_axis = np.int8(user_input);    
colors = ['blue','red','green']; # colors for target values: setosa blue, versicolor red, virginica green
fig = plt.figure(1);
plt.title('Scatterplot for iris dataset');
plt.xlabel(iris.feature_names[x_axis]);
plt.ylabel(iris.feature_names[y_axis]);
plt.scatter(iris.data[:,x_axis],iris.data[:,y_axis],s=50,c=iris.target,cmap=col.ListedColormap(colors));
plt.show();

# Scatterplot of the first two most important features
feature_selection = SelectKBest(k=2);
feature_selection.fit(iris.data,iris.target);
scores = feature_selection.scores_;
features = feature_selection.transform(iris.data);
mask = feature_selection.get_support();
feature_indices = [];
for i in range(p):
    if mask[i] == True : feature_indices.append(i);
x_axis, y_axis = feature_indices;

print('Importance weight of input attributes')
for i in range(p):
    print(iris.feature_names[i],': ',scores[i]);
fig = plt.figure(2);
plt.title('Scatterplot for iris dataset');
plt.xlabel(iris.feature_names[x_axis]);
plt.ylabel(iris.feature_names[y_axis]);
plt.scatter(iris.data[:,x_axis],iris.data[:,y_axis],s=50,c=iris.target,cmap=col.ListedColormap(colors));
plt.show();    

# Matrix scatterplot of Iris
iris_df = load_iris(as_frame=True);
sns.set(style="ticks");
sns.pairplot(iris_df.frame, hue="target");

# Full PCA using scikit-learn
pca = PCA();
pca.fit(iris.data);

# Visualizing the variance ratio which measures the importance of PCs
fig = plt.figure(4);
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 

# Visualizing the cumulative ratio which measures the impact of first n PCs
fig = plt.figure(5);
plt.title('Cumulative explained variance ratio plot');
cum_var_ratio = np.cumsum(var_ratio);
x_pos = np.arange(len(cum_var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,cum_var_ratio, align='center', alpha=0.5);
plt.show(); 

# PCA with limited components
pca = PCA(n_components=2);
pca.fit(iris.data);
iris_pc = pca.transform(iris.data);
class_mean = np.zeros((k,p));
for i in range(k):
    class_ind = [iris.target==i][0].astype(int);
    class_mean[i,:] = np.average(iris.data, axis=0, weights=class_ind);
PC_class_mean = pca.transform(class_mean);    
full_mean = np.reshape(pca.mean_,(1,4));
PC_mean = pca.transform(full_mean);

fig = plt.figure(6);
plt.title('Dimension reduction of the Iris data by PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(iris_pc[:,0],iris_pc[:,1],s=50,c=iris.target,
            cmap=col.ListedColormap(colors),label='Datapoints');
plt.scatter(PC_class_mean[:,0],PC_class_mean[:,1],s=50,marker='P',
            c=np.arange(k),cmap=col.ListedColormap(colors),label='Class means');
plt.scatter(PC_mean[:,0],PC_mean[:,1],s=50,c='black',marker='X',label='Overall mean');
plt.legend();
plt.show();
