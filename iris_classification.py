# ============================================================
# AI Tools Assignment - Part 2: Task 1
# Classical Machine Learning with Scikit-learn
# Dataset: Iris Species
# 1. Import libraries

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st

# 2. Load the dataset
# The Iris dataset is built into scikit-learn. It contains 150 samples
# of three Iris species with four features (sepal length, sepal width,
# petal length, and petal width).

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print("Sample of dataset:")
print(X.head())
print("\nTarget labels:", iris.target_names)

# ------------------------------
# 3. Data preprocessing

print("\nMissing values per column:")
print(X.isnull().sum())

# ------------------------------
# 4. Split the dataset

# We split the dataset into 80% training and 20% testing.
# This allows us to evaluate the model on unseen data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 5. Initialize and train model

# We’ll use a Decision Tree Classifier, which is a simple but powerful
# supervised ML algorithm. It's interpretable and works well on small datasets.

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 6. Make predictions

y_pred = model.predict(X_test)

# ------------------------------
# 7. Evaluate model performance

# We use accuracy, precision, and recall to assess the model.
# Precision and recall are computed with 'macro' average since we have multiple classes.

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# ------------------------------
# 8. Visualize confusion matrix

# A confusion matrix helps visualize how well the model predicted each class.

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Decision Tree Classifier")
st.pyplot(plt)


# ------------------------------
# 9. Visualize decision tree
# This shows how the tree splits data based on feature thresholds.

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Structure")
st.pyplot(plt)

# ------------------------------
# 10. Reflection

# Why this model works:
# - The Iris dataset is small and clean, making it ideal for tree-based models.
# - Decision trees can easily capture non-linear relationships between features.
# - Scikit-learn simplifies model training and evaluation through built-in methods.

# Limitations:
# - Decision trees can overfit small datasets.
# - Pruning or using ensemble methods (e.g., Random Forest) can improve generalization.

# ============================================================
# End of Task 1
# ============================================================
