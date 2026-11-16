
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("animal_data.csv")

print("Dataset Loaded Successfully!")

# 2. Select input features (X)
X = df[
    [
        "Weight",
        "Height",
        "Legs",
        "Habitat",
        "Diet",
        "Body_Covering"
    ]
]

# 3. Select target variable (y)
y = df["Animal_Class"]

# 4. Display input features
print("\nInput Features (X):")
print(X.head())

# 5. Display target variable
print("\nTarget Variable (y):")
print(y.head())

# 6. Display feature names
print("\nFeature Names:")
print(X.columns.tolist())

# 7. Display dataset dimensions
print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# 8. Display unique animal classes
print("\nAnimal Classes:")
print(y.unique())

# 9. Save features and target
X.to_csv("animal_features.csv", index=False)
y.to_csv("animal_target.csv", index=False)

print("\nFeature and Target Selection Completed Successfully!")
