# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:05:41 2019

Task: Fitting linear regression model for diabetes dataset
Results: regression model, prediction and graphical comparisons

Python tools    
Libraries: numpy, matplotlib, sklearn
Modules: random, special, pyplot, colors, linear_model
Classes: LinearRegression
Functions: load_diabetes, train_test_split

@author: Márton Ispány
"""

import numpy as np;   # Numerical Python library
import matplotlib.pyplot as plt;   # Matlab-like Python module
from sklearn.datasets import load_diabetes;  # importing dataset loader
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.linear_model import LinearRegression;  # importing linear regression class

diabetes = load_diabetes();
n = diabetes.data.shape[0];
p = diabetes.data.shape[1];

# Printing the basic parameters
print(f'Number of records:{n}');
print(f'Number of attributes:{p}');

# Printing a data value
# Deafult
record = 10;
feature = 2;
# Enter axis from consol
user_input = input('X axis [0..441, default:10]: ');
if len(user_input) != 0 and np.int16(user_input)>=0 and np.int16(user_input)<n :
    record = np.int16(user_input);
user_input = input('Y axis [0..9, default:2]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<p :
    feature = np.int8(user_input); 
print(diabetes.feature_names[feature],'[',record,']:', diabetes.data[record,feature]); 

# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, 
                                                    test_size=0.2, random_state=2020)

# Fitting linear regression
reg = LinearRegression();  # instance of the LinearRegression class
reg.fit(X_train,y_train);   #  fitting the model to data
intercept = reg.intercept_;  #  intecept (constant) parameter
coef = reg.coef_;    #  regression coefficients (weights)
score_train = reg.score(X_train,y_train);   #  R-square for goodness of fit
score_test = reg.score(X_test,y_test);
y_test_pred = reg.predict(X_test);   # prediction for test dataset

# Comparison of true and predicted target values  
plt.figure(1);
plt.title('Diabetes prediction');
plt.xlabel('True disease progression');
plt.ylabel('Predicted disease progression');
plt.scatter(y_test,y_test_pred,color="blue");
plt.plot([50,350],[50,350],color='red');
plt.show(); 

# Prediction for whole dataset
pred = reg.predict(diabetes.data);  # prediction by sklearn
pred1 = intercept*np.ones((n))+np.dot(diabetes.data,coef);  # prediction by numpy
error = diabetes.target-pred1;  # error of prediction
centered_target = diabetes.target-diabetes.target.mean(); 
score = reg.score(diabetes.data, diabetes.target);  # computing R-square by sklearn
score1 = 1-np.dot(error,error)/np.dot(centered_target,centered_target); # computing R-square by numpy
# Compare the last two value!


