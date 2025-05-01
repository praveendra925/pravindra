from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_deep = DecisionTreeClassifier(max_depth=None, random_state=42)
clf_deep.fit(X_train, y_train)
clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_pruned.fit(X_train, y_train)

print("=== Overfitting Tree ===")
print("Train Accuracy:", accuracy_score(y_train, clf_deep.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, clf_deep.predict(X_test)))
print("\n=== Pruned Tree (max_depth=3) ===")
print("Train Accuracy:", accuracy_score(y_train, clf_pruned.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, clf_pruned.predict(X_test)))
