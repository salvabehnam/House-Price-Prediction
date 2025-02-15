import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

# Save to CSV
df.to_csv("data/california_housing.csv", index=False)

print("Dataset saved successfully!")
