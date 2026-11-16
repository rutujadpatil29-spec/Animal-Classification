
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Load testing dataset
test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop("Animal_Class", axis=1)
y_test = test_data["Animal_Class"]

# Load trained models
decision_tree = joblib.load("decision_tree_model.pkl")
random_forest = joblib.load("random_forest_model.pkl")

# Generate predictions
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)

# Function to evaluate model
def evaluate_model(name, actual, predicted):

    print("\n==========", name, "==========")

    print("Accuracy:",
          accuracy_score(actual, predicted))

    print("Precision:",
          precision_score(actual, predicted,
                          average="weighted", zero_division=0))

    print("Recall:",
          recall_score(actual, predicted,
                       average="weighted", zero_division=0))

    print("F1 Score:",
          f1_score(actual, predicted,
                   average="weighted", zero_division=0))

    print("\nClassification Report:")
    print(classification_report(actual, predicted,
                                zero_division=0))

# Evaluate Decision Tree
evaluate_model("Decision Tree", y_test, dt_predictions)

# Evaluate Random Forest
evaluate_model("Random Forest", y_test, rf_predictions)

# Display confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, dt_predictions, ax=axes[0], cmap="Blues"
)
axes[0].set_title("Decision Tree Confusion Matrix")

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_predictions, ax=axes[1], cmap="Greens"
)
axes[1].set_title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\nConfusion Matrix Saved Successfully!")
print("\nModel Evaluation Completed Successfully!")
