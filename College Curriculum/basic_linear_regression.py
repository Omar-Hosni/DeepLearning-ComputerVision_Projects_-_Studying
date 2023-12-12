# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:01:42 2020

Task: Basic linear regression for simulated data
Input: slope and intercept parameters, sample size, error std
Ouput: estimated parameters R2 and plots

Python tools    
Libraries: numpy, matplotlib, sklearn
Modules: random, pyplot, linear_model
Classes: LinearRegression

@author: Márton Ispány
"""

import numpy as np;   # Numerical Python library
import matplotlib.pyplot as plt;  # Matlab-like Python module
from sklearn.linear_model import LinearRegression;  # Class for linear regression

# Default parameters
n = 1000;  # sample size
b0 = 3;   # intercept
b1 = 2;   # slope
sigma = 1; # error 

# Enter parameters from consol
user_input = input('Slope of regression line [default:2]: ');
if len(user_input) != 0 :
    b1 = np.float64(user_input);
user_input = input('Intercept of regression line [default:3]: ');
if len(user_input) != 0 :
    b0 = np.float64(user_input);
user_input = input('Sample size [default:1000]: ');
if len(user_input) != 0 :
    n = np.int64(user_input);
user_input = input('Error standard deviation [default:1]: ');
if len(user_input) != 0 :
    sigma = np.float64(user_input);

#  Generating random sample  
x = np.random.normal(0, 1, n);   #  standard normally distributed input
eps = np.random.normal(0, sigma, n);  #  random error
y = b0 + b1*x + eps;   #  regression equation

# Scatterplot for the first 100 records with regression line 
n_point = min(100,n);
plt.figure(1);
plt.title('Scatterplot of data with regression line');
plt.xlabel('x input');
plt.ylabel('y output');
xmin = min(x)-0.3;
xmax = max(x)+0.3;
ymin = b0 + b1*xmin;
ymax = b0 + b1*xmax;
plt.scatter(x[0:n_point],y[0:n_point],color="blue");  #  scatterplot of data
plt.plot([xmin,xmax],[ymin,ymax],color='red');  #  plot of regression line
plt.show(); 

# Fitting linear regression
reg = LinearRegression();  # instance of the LinearRegression class
X = x.reshape(1, -1).T;  # reshaping 1D array to 2D one
reg.fit(X,y);   #  fitting the model to data
b0hat = reg.intercept_;  #  estimated intercept
b1hat = reg.coef_[0];   #  estimated slope
R2 = reg.score(X,y);   #  R-square for model fitting
y_pred = reg.predict(X);  #  prediction of the target

# Computing the regression coefficients by using basic numpy
# Compare estimates below with b0hat and b1hat
reg_coef = np.ma.polyfit(x, y, 1);  

# Printing the results
print(f'Estimated slope:{b1hat:6.4f} (True slope:{b1})');
print(f'Estimated intercept:{b0hat:6.4f} (True intercept:{b0})');
print(f'R-square for goodness of fit:{R2:6.4f}');

# Scatterplot for data with true and estimated regression line
plt.figure(2);
plt.title('Scatterplot of data with regression lines');
plt.xlabel('x input');
plt.ylabel('y output');
xmin = min(x)-0.3;
xmax = max(x)+0.3;
ymin = b0 + b1*xmin;
ymax = b0 + b1*xmax;
plt.scatter(x[0:n_point],y[0:n_point],color="blue");
plt.plot([xmin,xmax],[ymin,ymax],color='black');
ymin = b0hat + b1hat*xmin;
ymax = b0hat + b1hat*xmax;
plt.plot([xmin,xmax],[ymin,ymax],color='red');
plt.show(); 

# Scatterplot for target prediction
n_point = min(1000,n);
plt.figure(3);
plt.title('Scatterplot for prediction');
plt.xlabel('True target');
plt.ylabel('Predicted target');
ymin = min(y)-1;
ymax = max(y)+1;
plt.scatter(y[0:n_point],y_pred[0:n_point],color="blue");
plt.plot([ymin,ymax],[ymin,ymax],color='red');
plt.show(); 

# End of code

