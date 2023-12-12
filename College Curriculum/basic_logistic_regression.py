# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:10:39 2020

Task: Basic logistic regression for simulated data
Input: slope and intercept parameters, sample size
Ouput: estimated parameters R2 and plots

Python tools    
Libraries: numpy, scipy, matplotlib, sklearn
Modules: random, special, pyplot, colors, linear_model
Classes: LogisticRegression

@author: Márton
"""

import numpy as np;  # Numerical Python library
import scipy as sp;  # Scientific Python library
import matplotlib.pyplot as plt;  # Matlab-like Python module
import matplotlib.colors as clr;  # importing coloring tools from MatPlotLib
from sklearn.linear_model import LogisticRegression; # Class for logistic regression


# Default parameters
n = 1000;  # sample size
b0 = 2;   # intercept
b1 = 3;   # slope

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

#  Generating random sample  
x = np.random.normal(0, 1, n); #  standard normally distributed input
z = b0 + b1*x;   # regression equation for latent variable
p = sp.special.expit(z);  # logistic transformation of latent variable using special function module
y = np.random.binomial(1,p);  # generating of random target from latent probability
# from latent variable by Bernoulli (Binomial(1,p)) random generator using random module

# Scatterplot for data
n_point = min(100,n);
plt.figure(1);
plt.title('Scatterplot with regression line');
plt.xlabel('x input variable');
plt.ylabel('z latent variable');
xmin = min(x)-0.3;
xmax = max(x)+0.3;
zmin = b0 + b1*xmin;
zmax = b0 + b1*xmax;
plt.scatter(x[0:n_point],z[0:n_point],color='blue');
plt.plot([xmin,xmax],[zmin,zmax],color='black');
plt.show(); 

# The logistic function
plt.figure(2);
plt.title('Logistic function');
plt.xlabel('x');
plt.ylabel('f(x)');
res = 0.0001;  #  resolution of the graph
base = np.arange(-5,5,res);
plt.scatter(base,sp.special.expit(base),s=5,color="blue");
plt.show(); 

# Scatterplot for data
plt.figure(3);
plt.title('Scatterplot for data with latent probabilities');
plt.xlabel('x input');
plt.ylabel('y output');
colors = ['blue','red'];
plt.scatter(x,p,color="black");
plt.scatter(x,y,c=y,cmap=clr.ListedColormap(colors));
plt.show(); 

# Fitting logistic regression
logreg = LogisticRegression();  # instance of the LogisticRegression class
X = x.reshape(1, -1).T;
logreg.fit(X,y);  #  fitting the model to data
b0hat = logreg.intercept_[0];  #  estimated intercept
b1hat = logreg.coef_[0,0];  #  estimated slope
accuracy = logreg.score(X,y);  #  accuracy for model fitting
y_pred_logreg = logreg.predict(X);  #  prediction of the target
p_pred_logreg = logreg.predict_proba(X);  # posterior distribution for the target

# Printing the results
print(f'Estimated slope:{b1hat:6.4f} (True slope:{b1})');
print(f'Estimated intercept:{b0hat:6.4f} (True intercept:{b0})');
print(f'Accuracy:{accuracy:6.4f}');

# Scatterplot for latent probabilities
plt.figure(4);
n_point = min(1000,n);
plt.title('Scatterplot for fitting latent probabilities');
plt.xlabel('p true');
plt.ylabel('p estimated');
plt.scatter(p[0:n_point],p_pred_logreg[0:n_point,1],color='blue');
plt.plot([0,1],[0,1],color='black');
plt.show(); 

# Computing the latent variable z as decision function: 
# the predicted target is 1 if y>0 and 0 if y<0
z_pred = logreg.decision_function(X);
# Predicition of latent variable z by logit transformation
# Compare the above values
z_pred1 = sp.special.logit(p_pred_logreg[:,1]);  

# Scatterplot for latent variable z
plt.figure(5);
n_point = min(1000,n);
plt.title('Scatterplot for latent variable');
plt.xlabel('z true');
plt.ylabel('z estimated');
plt.scatter(z[0:n_point],z_pred[0:n_point],color='blue');
zmin = min(z)-0.3;
zmax = max(z)+0.3;
zmin1 = min(z_pred)-0.3;
zmax1 = max(z_pred)+0.3;
plt.plot([zmin,zmax],[zmin1,zmax1],color='black');
plt.show(); 

# End of code

