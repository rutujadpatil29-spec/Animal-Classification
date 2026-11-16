
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load testing dataset
test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop("Animal_Class", axis=1)
y_test = test_data["Animal_Class"]

# Load trained models
decision_tree = joblib.load("decision_tree_model.pkl")
random_forest = joblib.load("random_forest_model.pkl")

# Generate predictions
dt_pred = decision_tree.predict(X_test)
rf_pred = random_forest.predict(X_test)

# Calculate performance metrics
def calculate_metrics(actual, predicted):

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted,
            average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted,
            average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            actual, predicted,
            average="weighted", zero_division=0
        )
    }

# Compare both models
comparison = pd.DataFrame(
    {
        "Decision Tree": calculate_metrics(y_test, dt_pred),
        "Random Forest": calculate_metrics(y_test, rf_pred)
    }
)

print("\n========== MODEL COMPARISON ==========")
print(comparison)

# Identify the model with higher accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

if dt_accuracy > rf_accuracy:
    best_model = "Decision Tree"
elif rf_accuracy > dt_accuracy:
    best_model = "Random Forest"
else:
    best_model = "Both models have equal accuracy"

print("\nBest Model Based on Accuracy:", best_model)

# Save comparison results
comparison.to_csv("model_comparison.csv")

print("\nComparison Results Saved Successfully!")
print("\nModel Comparison Completed Successfully!")
