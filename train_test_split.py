
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("animal_data.csv")

print("Dataset Loaded Successfully!")
print("Total Records:", len(df))

# 2. Select features (X)
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

# 3. Select target (y)
y = df["Animal_Class"]

# 4. Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Display dataset sizes
print("\nTraining and Testing Dataset:")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 6. Display class distribution
print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

# 7. Save training and testing datasets
train_data = X_train.copy()
train_data["Animal_Class"] = y_train
train_data.to_csv("train_data.csv", index=False)

test_data = X_test.copy()
test_data["Animal_Class"] = y_test
test_data.to_csv("test_data.csv", index=False)

print("\nTrain-Test Split Completed Successfully!")
