from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))
