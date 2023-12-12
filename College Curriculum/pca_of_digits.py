# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:18:03 2020

Principal Component (PC) analysis of digits dataset

@author: Márton Ispány
"""

import numpy as np;  # importing numerical library
from matplotlib import pyplot as plt;  # importing the MATLAB-like plotting tool
from sklearn.datasets import load_digits; # importing digit dataset
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.decomposition import PCA;  # importing PCA


# loading dataset and computing dimensions
digits = load_digits();
n = digits.data.shape[0]; # number of records
p = digits.data.shape[1]; # number of attributes

# Visualizing digit images 
# Default index
image_ind = 10;  #  index of the image
user_input = input('Image index [0..1796, default:10]: ');
if len(user_input) != 0 and np.int16(user_input)>=0 and np.int16(user_input)<n :
    image_ind = np.int16(user_input);
plt.matshow(15-digits.images[image_ind]);
plt.show();

# Partitioning into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, 
             digits.target, test_size=0.3, random_state=2020);

# Full PCA on training set
pca = PCA();
pca.fit(X_train);

# Visualizing the variance ratio which measures the importance of principal components
fig = plt.figure(2);
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio))+1;
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 

# Visualizing the cumulative ratio which measures the impact of first n PCs
fig = plt.figure(3);
plt.title('Cumulative explained variance ratio plot');
cum_var_ratio = np.cumsum(var_ratio);
x_pos = np.arange(len(cum_var_ratio))+1;
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,cum_var_ratio, align='center', alpha=0.5);
plt.show(); 

# Visualizing the training set in 2D PC space by using colors for different digits
PC_train = pca.transform(X_train);
fig = plt.figure(4);
plt.title('Scatterplot for training digits dataset');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_train[:,0],PC_train[:,1],s=50,c=y_train,cmap = 'tab10');
plt.show();

# Visualizing the test set in 2D PC space
PC_test = pca.transform(X_test);
fig = plt.figure(5);
plt.title('Scatterplot for test digits dataset');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_test[:,0],PC_test[:,1],s=50,c=y_test,cmap = 'tab10');
plt.show();

# Compare the last two figures! 