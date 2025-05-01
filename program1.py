from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)

# Plot the Decision Tree using matplotlib
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, rounded=True)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()

