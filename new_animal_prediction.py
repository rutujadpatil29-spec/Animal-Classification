
import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

print("Animal Classification System")
print("-----------------------------")

# Enter new animal details
weight = float(input("Enter Weight (kg): "))
height = float(input("Enter Height (cm): "))
legs = int(input("Enter Number of Legs: "))

habitat = input(
    "Enter Habitat (Land/Water/Air): "
).strip().capitalize()

diet = input(
    "Enter Diet (Herbivore/Carnivore/Omnivore): "
).strip().capitalize()

body_covering = input(
    "Enter Body Covering (Fur/Feathers/Scales/Skin): "
).strip().capitalize()

# Create input DataFrame
new_animal = pd.DataFrame([{
    "Weight": weight,
    "Height": height,
    "Legs": legs,
    "Habitat": habitat,
    "Diet": diet,
    "Body_Covering": body_covering
}])

# Predict animal class
prediction = model.predict(new_animal)

print("\n========== PREDICTION RESULT ==========")
print("Predicted Animal Class:", prediction[0])

print("\nPrediction Completed Successfully!")
