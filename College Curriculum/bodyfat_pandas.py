# -*- coding: utf-8 -*-
"""
Created on Mon Dec 6 10:33:05 2021

Task: importing data from url into dataframe with column name
computing descriptive stats and visualizing

Python tools    
Libraries: numpy, urllib, pandas, matplotlib
Modules: pyplot, request, plotting
Classes:  
Functions: urlopen, DataFrame, parallel_coordinates, scatter_matrix

@author: Márton Ispány
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
import pandas as pd;  # importing pandas data analysis tool
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework

# Reading the dataset
url = 'https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/bodyfat.csv';
raw_data = urlopen(url);
attribute_names = np.loadtxt(raw_data, max_rows=1, delimiter=",",dtype=np.str);  # reading the first row with attribute names
data = np.loadtxt(raw_data, delimiter=",");  # reading numerical data from csv file
del raw_data;
# Removing unnecessary "s and spaces from names
for i in range(len(attribute_names)):
    attribute_names[i] = attribute_names[i].replace('"','');
    attribute_names[i] = attribute_names[i].replace(' ','');

# Defining dataframes with column names from numpy array
df = pd.DataFrame(data=data, columns=attribute_names);  #  reading the data into dataframe

# Grouping by age and computing the mean
mean_by_age = df.groupby(by="Age").mean();
print(mean_by_age[["Density","Height","Weight"]]);

# Parallel axis graph by Age
plt.figure(1);
pd.plotting.parallel_coordinates(df,class_column='Age');
plt.show();

# Scatter matrix for partial columns
pd.plotting.scatter_matrix(df[["Density","Height","Weight"]]);

