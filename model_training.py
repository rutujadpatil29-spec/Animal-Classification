
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load training and testing datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

print("Training and Testing Data Loaded Successfully!")

# 2. Separate features and target
X_train = train_data.drop("Animal_Class", axis=1)
y_train = train_data["Animal_Class"]

X_test = test_data.drop("Animal_Class", axis=1)
y_test = test_data["Animal_Class"]

# 3. Identify numerical and categorical features
numerical_features = ["Weight", "Height", "Legs"]

categorical_features = [
    "Habitat",
    "Diet",
    "Body_Covering"
]

# 4. Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 5. Create Decision Tree Pipeline
decision_tree = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

# 6. Create Random Forest Pipeline
random_forest = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])

# 7. Train Decision Tree
print("\nTraining Decision Tree Model...")
decision_tree.fit(X_train, y_train)

# 8. Train Random Forest
print("Training Random Forest Model...")
random_forest.fit(X_train, y_train)

# 9. Generate predictions
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)

# 10. Display predictions
print("\nDecision Tree Predictions:")
print(dt_predictions[:10])

print("\nRandom Forest Predictions:")
print(rf_predictions[:10])

# 11. Save trained models
joblib.dump(decision_tree, "decision_tree_model.pkl")
joblib.dump(random_forest, "random_forest_model.pkl")

# 12. Save predictions
results = pd.DataFrame({
    "Actual_Animal": y_test.values,
    "Decision_Tree_Prediction": dt_predictions,
    "Random_Forest_Prediction": rf_predictions
})

results.to_csv("model_predictions.csv", index=False)

print("\nModels Saved Successfully!")
print("Predictions Saved Successfully!")
print("\nModel Training Completed Successfully!")
