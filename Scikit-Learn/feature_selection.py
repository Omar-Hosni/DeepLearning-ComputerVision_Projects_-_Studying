# Selection - SelectPercentile in Sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2


X,y = load_digits(return_X_y=True)

X_new = SelectPercentile(chi2, percentile=10).fit_transform(X,y)

