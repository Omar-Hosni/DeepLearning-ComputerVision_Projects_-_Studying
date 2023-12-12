# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:06:05 2020

Task: Replication analysis of basic linear regression
Results: descriptive stats and statistical graphs showing the randomness of model parameters

Python tools    
Libraries: numpy, matplotlib, sklearn
Modules: pyplot, random, linear_model, utils, model_selection
Classes: LinearRegression
Functions: normal, sample_without_replacement, hist, train_test_split, cross_validate

@author: Márton Ispány
"""

import numpy as np;   # importing numerical computing library
import matplotlib.pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.linear_model import LinearRegression;  # importing linear regression class
from sklearn.utils.random import sample_without_replacement;  #  importing sampling
from sklearn.model_selection import train_test_split;  # importing splitting
from sklearn.model_selection import cross_validate; # importing crossvalidation

# Default parameters
n = 100000;  # sample size
b0 = 3;   # intercept
b1 = 2;   # slope
sig = 1;  # error

# Enter parameters from consol
user_input = input('Slope of regression line [default:2]: ');
if len(user_input) != 0 :
    b1 = np.float64(user_input);
user_input = input('Intercept of regression line [default:3]: ');
if len(user_input) != 0 :
    b0 = np.float64(user_input);
user_input = input('Sample size [default:100000]: ');
if len(user_input) != 0 :
    n = np.int64(user_input);
user_input = input('Error standard deviation [default:1]: ');
if len(user_input) != 0 :
    sigma = np.float64(user_input);

#  Generating random sample  
x = np.random.normal(0, 1, n);  #  standard normally distributed input
eps = np.random.normal(0, sig, n);  #  random error
y = b0 + b1*x + eps;  #  regression equation

# Default replication parameters
rep = 100;  #  number of replications
sample_size = 1000;  # sample size
reg = LinearRegression();  # instance of the LinearRegression class
b0hat = [];  #  list for intercept
b1hat = [];  # list for slope
score = [];  # list for R-squares

for i in range(rep):
    # random sampling from dataset
    index = sample_without_replacement(n_population = n, n_samples = sample_size);
    x_sample = x[index];
    y_sample = y[index];
    X_sample = x_sample.reshape(1, -1).T;
    reg.fit(X_sample,y_sample);
    b0hat.append(reg.intercept_);
    b1hat.append(reg.coef_);
    score.append(reg.score(X_sample,y_sample));
    
b0hat_mean = np.mean(b0hat);
b0hat_std = np.std(b0hat);
b1hat_mean = np.mean(b1hat);
b1hat_std = np.std(b1hat);
score_mean = np.mean(score);
score_std = np.std(score);

# Printing the results
print(f'Mean slope:{b1hat_mean:6.4f} (True slope:{b1}) with standard deviation {b1hat_std:6.4f}');
print(f'Mean intercept:{b0hat_mean:6.4f} (True intercept:{b0}) with standard deviation {b0hat_std:6.4f}');
print(f'Mean of R-square for goodness of fit:{score_mean:6.4f} (standard deviation: {score_std:6.4f})');
    
# Histograms for parameters and scores 
plt.figure(1);
n, bins, patches = plt.hist(np.asarray(b1hat), bins=25, color='g', alpha=0.75);
plt.xlabel('Slope');
plt.ylabel('Frequency');
plt.title('Histogram of slope');
plt.text(b1-2.5*b1hat_std, 17, f'$\mu={b1hat_mean:4.3f},\ \sigma={b1hat_std:4.3f}$');
plt.xlim(b1-3*b1hat_std, b1+3*b1hat_std);
plt.grid(True);
plt.show();

plt.figure(2);
n, bins, patches = plt.hist(np.asarray(b0hat), bins=25, color='g', alpha=0.75);
plt.xlabel('Intercept');
plt.ylabel('Frequency');
plt.title('Histogram of intercept');
plt.text(b0-2.5*b1hat_std, 17, f'$\mu={b0hat_mean:4.3f},\ \sigma={b0hat_std:4.3f}$');
plt.xlim(b0-3*b1hat_std, b0+3*b1hat_std);
plt.grid(True);
plt.show();

plt.figure(3);
n, bins, patches = plt.hist(np.asarray(score), bins=25, color='g', alpha=0.75);
plt.xlabel('Score');
plt.ylabel('Frequency');
plt.title('Histogram of R-square score');
plt.text(0.77, 8, f'$\mu={score_mean:4.3f},\ \sigma={score_std:4.3f}$');
plt.xlim(score_mean-3*score_std, score_mean+3*score_std);
plt.grid(True);
plt.show();

# Above results clearly demonstrate the dependence of the parameter estimation of the training set
# As the training set comes from the data warehouse randomly the results of a machine learning process
# will be random

# Replication analysis for test dataset
# One training set, one parameter estimation
# Several test sets, distribution of score value
score = [];  # list for R-squares of test sets
# Fitting the model for the first
n_train = 10000;
x_sample = x[0:n_train];
y_sample = y[0:n_train];
X_sample = x_sample.reshape(1, -1).T;
reg.fit(X_sample,y_sample);
b0hat = reg.intercept_;
b1hat = reg.coef_;
score_train = reg.score(X_sample,y_sample);

for i in range(rep):
    # random sampling from dataset
    index = sample_without_replacement(n_population = n, n_samples = sample_size);
    x_test = x[index];
    y_test = y[index];
    X_test = x_test.reshape(1, -1).T;
    score.append(reg.score(X_test,y_test));
    
score_mean = np.mean(score);
score_std = np.std(score);

# Printing the results
print(f'Mean of test R-squares:{score_mean:6.4f} (standard deviation: {score_std:6.4f}), training R-square: {score_train:6.4f}');

# Histogram of test R-square scores with train one as red line
plt.figure(4);
n, bins, patches = plt.hist(np.asarray(score), bins=25, color='g', alpha=0.75);
plt.xlabel('Score');
plt.ylabel('Frequency');
plt.title('Histogram of test R-square score');
plt.text(0.765, 8, f'$\mu={score_mean:4.3f},\ \sigma={score_std:4.3f}$');
plt.xlim(score_mean-3*score_std, score_mean+3*score_std);
plt.grid(True);
plt.vlines(score_train, 0, 14, colors='r')
plt.show();

# Splitting the dataset for training and test ones
X_train, X_test, y_train, y_test = train_test_split(x.reshape(1, -1).T,y, test_size=0.3, 
                                shuffle = True, random_state=2020);
reg.fit(X_train,y_train);
b0hat = reg.intercept_;
b1hat = reg.coef_[0];
score_train = reg.score(X_train,y_train);
score_test = reg.score(X_test,y_test);

# Printing the results
print(f'Estimated slope:{b1hat:6.4f} (True slope:{b1})');
print(f'Estimated intercept:{b0hat:6.4f} (True intercept:{b0})');
print(f'Training R-square:{score_train:6.4f}, Test R-square: {score_test:6.4f})');

# Crossvalidation of regression model
cv_results = cross_validate(reg, x.reshape(1, -1).T, y, cv=10);
score_mean = cv_results['test_score'].mean();
score_std = cv_results['test_score'].std();

# Printing the results
print(f'Mean of R-square in crossvalidation:{score_mean:6.4f} (standard deviation: {score_std:6.4f})');
  
