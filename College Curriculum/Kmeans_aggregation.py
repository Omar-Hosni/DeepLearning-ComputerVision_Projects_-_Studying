# -*- coding: utf-8 -*-
"""
Created on Mon May 4 00:00:55 2020

Task: Clustering of Aggregation dataset from the URL
https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Clustering/

Python tools    
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, cluster, metrics
Classes: KMeans
Functions: urlopen, davies_bouldin_score

@author: Márton Ispány
"""

import numpy as np;  # Numerical Python library
from matplotlib import pyplot as plt;  # Matlab-like Python module
from urllib.request import urlopen;  # importing url handling
from sklearn.cluster import KMeans;  # importing clustering algorithms
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit

url = 'https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Clustering/Aggregation.tsv';
raw_data = urlopen(url);  # opening url
data = np.loadtxt(raw_data, delimiter="\t");  # loading dataset
X = data[:,0:2];  #  input attributes
y = data[:,2];   #  label attribute

# Visualizing of datapoints using label colors
fig = plt.figure(1);
plt.title('Scatterplot of datapoints with labels');
plt.xlabel('X1');
plt.ylabel('X2');
plt.scatter(X[:,0],X[:,1],s=50,c=y);
plt.show();

# Default parameters
K = 7;

# Enter parameters from consol
user_input = input('Number of clusters [default:7]: ');
if len(user_input) != 0 :
    K = np.int8(user_input);

# K-means clustering with fix K
kmeans_cluster = KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);   #  fiting cluster model for X
y_pred = kmeans_cluster.predict(X);   #  predicting cluster label
sse = kmeans_cluster.inertia_;   # sum of squares of error (within sum of squares)
centers = kmeans_cluster.cluster_centers_;  # centroid of clusters

# Davies-Bouldin goodness-of-fit
DB = davies_bouldin_score(X,y_pred);  

# Printing the results
print(f'Number of cluster: {K}');
print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');

# Visualizing of datapoints with cluster labels and centroids
fig = plt.figure(2);
plt.title('Scatterplot of datapoints with clusters');
plt.xlabel('X1');
plt.ylabel('X2');
plt.scatter(X[:,0],X[:,1],s=50,c=y_pred);   #  dataponts with cluster label
plt.scatter(centers[:,0],centers[:,1],s=50,c='red');  #  centroids
plt.show();

# Finding optimal cluster number
Max_K = 31;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(X);
    y_pred = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(X,y_pred);

# Visualization of SSE values     
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

# The local minimum of Davies Bouldin curve gives the optimal cluster number
# The optimal cluster numbers are K = 4,6

# K-means clusterings and visualization with fix K
K = 4;
kmeans_cluster = KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);
y_pred = kmeans_cluster.predict(X);
centers = kmeans_cluster.cluster_centers_;

fig = plt.figure(5);
plt.title('Scatterplot of datapoints with 4 clusters');
plt.xlabel('X1');
plt.ylabel('X2');
plt.scatter(X[:,0],X[:,1],s=50,c=y_pred);
plt.scatter(centers[:,0],centers[:,1],s=50,c='red');
plt.show();

K = 6;
kmeans_cluster = KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);
y_pred = kmeans_cluster.predict(X);
centers = kmeans_cluster.cluster_centers_;

fig = plt.figure(6);
plt.title('Scatterplot of datapoints with 6 clusters');
plt.xlabel('X1');
plt.ylabel('X2');
plt.scatter(X[:,0],X[:,1],s=50,c=y_pred);
plt.scatter(centers[:,0],centers[:,1],s=50,c='red');
plt.show();

# End of code