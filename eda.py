
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load original dataset
df = pd.read_csv("animal_data.csv")

sns.set_theme(style="whitegrid")

# 1. Animal Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Animal_Class")
plt.title("Animal Class Distribution")
plt.xlabel("Animal Class")
plt.ylabel("Number of Animals")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("graph1_class_distribution.png")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(8, 5))
numeric_data = df[["Weight", "Height", "Legs"]]
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("graph2_correlation.png")
plt.show()

# 3. Weight vs Height Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Weight",
    y="Height",
    hue="Animal_Class"
)
plt.title("Weight vs Height of Animals")
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.tight_layout()
plt.savefig("graph3_scatter.png")
plt.show()

# 4. Weight Distribution Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Animal_Class",
    y="Weight"
)
plt.title("Weight Distribution by Animal Class")
plt.xlabel("Animal Class")
plt.ylabel("Weight (kg)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("graph4_boxplot.png")
plt.show()

print("All four EDA graphs created successfully!")
