# ============================================================
# AI Tools Assignment - Part 2: Task 1
# Classical Machine Learning with Scikit-learn
# Dataset: Iris Species
# ============================================================

# 1. Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 2. Page Setup
st.set_page_config(page_title="Iris Classification Model", layout="wide")
st.title("🌸 Iris Flower Classification App")
st.write("This app trains a Decision Tree model using Scikit-learn to classify iris flower species.")

# ============================================================
# 3. Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Display sample of dataset on webpage
st.subheader("Sample of Dataset")
st.dataframe(X.head())

# Show target labels
st.subheader("Target Labels")
st.write(iris.target_names)

# ============================================================
# 4. Data preprocessing
missing_values = X.isnull().sum()
st.subheader("Missing Values per Column")
st.write(missing_values)

# ============================================================
# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 6. Initialize and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ============================================================
# 7. Make predictions and evaluate model performance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")

# Display performance metrics on webpage
st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy:.3f}")
st.metric("Precision", f"{precision:.3f}")
st.metric("Recall", f"{recall:.3f}")

# ============================================================
# 8. Visualize confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

fig1, ax1 = plt.subplots()
disp.plot(cmap="Blues", values_format="d", ax=ax1)
st.pyplot(fig1)

# ============================================================
# 9. Visualize decision tree
st.subheader("Decision Tree Structure")
fig2, ax2 = plt.subplots(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, ax=ax2)
st.pyplot(fig2)

# ============================================================
# 10. Reflection
st.subheader("Reflection")
st.markdown("""
**Why this model works well:**
- The Iris dataset is clean and small — ideal for a decision tree.
- Decision trees handle non-linear relationships and are interpretable.
- Scikit-learn makes training and evaluating models simple.

**Limitations:**
- Trees can overfit small datasets.
- Ensemble methods like Random Forest can improve performance.
""")

# ============================================================
# End of Task 1
# ============================================================
