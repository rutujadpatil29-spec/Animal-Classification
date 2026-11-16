
import pandas as pd

# 1. Load dataset
df = pd.read_csv("animal_data.csv")

print("\n========== ANIMAL DATASET ==========")

# 2. Display first five records
print("\n1. First Five Records:")
print(df.head())

# 3. Dataset dimensions
print("\n2. Dataset Dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 4. Identify data types
print("\n3. Data Types:")
print(df.dtypes)

# 5. Identify missing values
print("\n4. Missing Values:")
print(df.isnull().sum())

# 6. Identify duplicate records
print("\n5. Duplicate Records:")
print("Total Duplicates:", df.duplicated().sum())

# 7. Numerical features
print("\n6. Numerical Features:")
print(df.select_dtypes(include=["number"]).columns.tolist())

# 8. Categorical features
print("\n7. Categorical Features:")
print(df.select_dtypes(include=["object"]).columns.tolist())

# 9. Dataset information
print("\n8. Dataset Information:")
df.info()

# 10. Animal class distribution
print("\n9. Animal Class Distribution:")
print(df["Animal_Class"].value_counts())

print("\nDataset Understanding Completed Successfully!")
