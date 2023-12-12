import os
import time
import dask
import numpy as np
import dask.dataframe as dd
from sklearn.svm import SVC
from dask.distributed import Client
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV


def sequential_approach():
    # Generate a larger dataset
    X, y = make_classification(n_samples=10000,
                               n_features=20,
                               random_state=42)
    X = dd.from_array(X, chunksize=100)
    y = dd.from_array(y, chunksize=100)

    # Define the SVM model and hyperparameters
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    }

    svm = SVC()

    # Perform hyperparameter tuning sequentially
    start_time = time.time()
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X.compute(), y.compute())
    end_time = time.time()

    # Evaluate the best SVM model
    best_svm = grid_search.best_estimator_
    accuracy = best_svm.score(X.compute(), y.compute())

    print("Sequential Approach - Best SVM Model:", best_svm)
    print("Accuracy:", accuracy)
    print("Execution Time (Sequential):", end_time - start_time)

def parallel_approach():

    # Generate a larger dataset
    X, y = make_classification(n_samples=1000,
                               n_features=20,
                               random_state=42)
    X = dd.from_array(X, chunksize=1000)
    y = dd.from_array(y, chunksize=1000)

    # Convert Dask DataFrames to Pandas DataFrames
    X_pd = X.compute()
    y_pd = y.compute()

    # Define the SVM model and hyperparameters
    param_grid = {
         'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    }

    svm = SVC()

    # Perform hyperparameter tuning in parallel using Dask
    start_time = time.time()
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_pd, y_pd)
    end_time = time.time()

    # Evaluate the best SVM model
    best_svm = grid_search.best_estimator_
    accuracy = best_svm.score(X_pd, y_pd)

    print("Parallel Approach - Best SVM Model:", best_svm)
    print("Accuracy:", accuracy)
    print("Execution Time (Parallel):", end_time - start_time)


if __name__ == "__main__":
    parallel_approach()
