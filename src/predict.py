import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest_model.pkl")

# Sample input
sample_data = pd.DataFrame([[8.3252, 41, 6.984, 1.023, 322, 2.555, 37.88, -122.23]],
                           columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"])

# Make prediction
prediction = model.predict(sample_data)

print(f"Predicted House Price: {prediction[0]:.2f}")
