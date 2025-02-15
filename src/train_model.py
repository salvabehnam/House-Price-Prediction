import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("data/california_housing.csv")  # Adjust if needed

# Split dataset
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

# Generate feature importance visualization
feature_importance = model.feature_importances_
features = X.columns

# Create a bar plot
plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=feature_importance)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest Model")
plt.xticks(rotation=45)

# Save the visualization
plt.savefig("visuals/feature_importance.png")  # ✅ Saves the plot inside visuals/
plt.show()

print(f"Model trained and saved! MSE: {mse:.4f}")
print("Feature importance plot saved to 'visuals/feature_importance.png'.")
