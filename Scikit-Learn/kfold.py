from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

kf = KFold(n_splits=2)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # SVM classifier
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_accuracy = svm.score(X_test, y_test)
    print("SVM Accuracy:", svm_accuracy)

    # Decision Tree classifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree_accuracy = tree.score(X_test, y_test)
    print("Decision Tree Accuracy:", tree_accuracy)

