
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Create Decision Tree model
decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

# 2. Create Random Forest model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

# 3. Store models
models = {
    "Decision Tree": decision_tree,
    "Random Forest": random_forest
}

# 4. Display selected models
print("Selected Machine Learning Models:")

for name, model in models.items():
    print("\nModel:", name)
    print(model)

print("\nModel Selection Completed Successfully!")
