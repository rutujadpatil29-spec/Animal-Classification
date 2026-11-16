
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load dataset
df = pd.read_csv("animal_data.csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:", df.shape)

# 2. Handle missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill numerical missing values with median
numerical_cols = df.select_dtypes(include=["number"]).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. Remove duplicate records
print("\nDuplicates Before Removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates After Removal:", df.duplicated().sum())

# 4. Encode categorical variables
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nDataset After Encoding:")
print(df.head())

# 5. Detect outliers using IQR method
print("\nOutlier Detection:")

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(col, "Outliers:", len(outliers))

# 6. Feature scaling
feature_cols = ["Weight", "Height", "Legs"]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("\nDataset After Feature Scaling:")
print(df.head())

# 7. Save processed dataset
df.to_csv("processed_animal_data.csv", index=False)

print("\nProcessed Dataset Shape:", df.shape)

print("\nData Preprocessing Completed Successfully!")
