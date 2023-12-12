# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:55:30 2020

Task: Assessing of classifiers fitted for Spambase dataset
Binary (Binomial) classification problem
Classifiers: logistic regression, naive Bayes
Results: confusion matrix, ROC curve, AUC value
Original data source: https://archive.ics.uci.edu/ml/datasets/spambase

Python tools    
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, linear_model, naive_bayes, model_selection, metrics
Classes: LogisticRegression, GaussianNB
Functions: urlopen, train_test_split, confusion_matrix, roc_curve, auc

@author: Márton Ispány
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve; #  importing performance metrics

    
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

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);   # spam prediction for train
accuracy_logreg_train = logreg_classifier.score(X_train,y_train);
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction for test
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities
accuracy_logreg_test = logreg_classifier.score(X_test,y_test);

# Plotting non-normalized confusion matrix
plot_confusion_matrix(logreg_classifier, X_train, y_train, display_labels = target_names);

plot_confusion_matrix(logreg_classifier, X_test, y_test, display_labels = target_names);

# Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);  # spam prediction for train
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities

# Plotting non-normalized confusion matrix
plot_confusion_matrix(naive_bayes_classifier, X_train, y_train, display_labels = target_names);

plot_confusion_matrix(naive_bayes_classifier, X_test, y_test, display_labels = target_names); 

# Plotting ROC curve
plot_roc_curve(logreg_classifier, X_test, y_test);
plot_roc_curve(naive_bayes_classifier, X_test, y_test);

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, yprobab_naive_bayes[:,1]);
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(7);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

# Another method for visualizing confusion matrix

import itertools;
#  definition of plotting function
def plot_cm(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure(8);
plot_cm(cm_logreg_train, classes=target_names,
        title = 'Confusion matrix for training dataset (logistic regression)');
plt.show();

plt.figure(9);
plot_cm(cm_logreg_test, classes=target_names,
   title='Confusion matrix for test dataset (logistic regression)');
plt.show();

plt.figure(10);
plot_cm(cm_naive_bayes_train, classes=target_names,
    title='Confusion matrix for training dataset (naive Bayes)');
plt.show();

plt.figure(11);
plot_cm(cm_naive_bayes_test, classes=target_names,
   title='Confusion matrix for test dataset (naive Bayes)');
plt.show();