from sklearn.model_selection import KFold
from itertools import product
from sklearn.base import clone

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None


    def fit(self, X, y):
        param_combinations = list(product(*self.param_grid.values()))
        num_combinations = len(param_combinations)
        cv_results = []

        for i, params in enumerate(param_combinations):
            self.update_estimator_params(params)
            scores = []
            best_score_per_fold = []

            kf = KFold(n_splits=self.cv)

            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                cloned_estimator = clone(self.estimator)
                cloned_estimator.fit(X_train, y_train)

                if self.scoring:
                    score = self.scoring(cloned_estimator, X_val, y_val)
                else:
                    score = cloned_estimator.score(X_val, y_val)

                scores.append(score)
                best_score_per_fold.append(score)

            average_score = sum(scores) / self.cv
            cv_results.append((params, best_score_per_fold))

            if average_score > self.best_score_:
                self.best_score_ = average_score
                self.best_params_ = params
                self.best_estimator_ = clone(self.estimator)

            print(f'progess:{i+1}/{num_combinations}')

        self.cv_results_ = cv_results

    def update_estimator_params(self, params):
        for key, value in zip(self.param_grid.keys(), params):
            setattr(self.estimator, key, value)