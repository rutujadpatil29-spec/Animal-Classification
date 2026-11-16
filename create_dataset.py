
import csv
import random

random.seed(42)

data = []

animal_types = {
    "Mammal": {
        "weight": (1, 500),
        "height": (10, 300),
        "legs": [2, 4],
        "habitat": ["Land", "Water"],
        "diet": ["Herbivore", "Carnivore", "Omnivore"],
        "covering": ["Fur", "Hair"]
    },
    "Bird": {
        "weight": (0.05, 20),
        "height": (10, 200),
        "legs": [2],
        "habitat": ["Air", "Land"],
        "diet": ["Herbivore", "Carnivore", "Omnivore"],
        "covering": ["Feathers"]
    },
    "Reptile": {
        "weight": (0.1, 100),
        "height": (5, 250),
        "legs": [0, 4],
        "habitat": ["Land", "Water"],
        "diet": ["Carnivore", "Herbivore"],
        "covering": ["Scales"]
    },
    "Fish": {
        "weight": (0.01, 100),
        "height": (2, 150),
        "legs": [0],
        "habitat": ["Water"],
        "diet": ["Carnivore", "Herbivore", "Omnivore"],
        "covering": ["Scales"]
    },
    "Amphibian": {
        "weight": (0.01, 5),
        "height": (2, 40),
        "legs": [0, 4],
        "habitat": ["Land", "Water"],
        "diet": ["Carnivore", "Omnivore"],
        "covering": ["Moist Skin"]
    }
}

# Generate 60 records for each animal class
for animal_class, features in animal_types.items():

    for i in range(60):

        weight = round(random.uniform(*features["weight"]), 2)
        height = round(random.uniform(*features["height"]), 2)
        legs = random.choice(features["legs"])
        habitat = random.choice(features["habitat"])
        diet = random.choice(features["diet"])
        covering = random.choice(features["covering"])

        data.append([
            weight,
            height,
            legs,
            habitat,
            diet,
            covering,
            animal_class
        ])

# Shuffle dataset
random.shuffle(data)

# Save dataset as CSV
columns = [
    "Weight",
    "Height",
    "Legs",
    "Habitat",
    "Diet",
    "Body_Covering",
    "Animal_Class"
]

with open("animal_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    writer.writerows(data)

print("Animal dataset created successfully!")
print("Total records:", len(data))
print("Total features:", len(columns) - 1)
