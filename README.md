# 🏡 House Price Prediction using Machine Learning
This project builds a **machine learning model** to predict house prices using **Random Forest & Linear Regression**.  
We use the **California Housing Dataset** from Scikit-learn and evaluate performance with **MSE & R² Score**.

## 📌 Features
- **Feature Engineering & Data Preprocessing**
- **Trained Random Forest Model (R² = 0.8)**
- **Feature Importance Visualization**
- **Modular Code (Train, Predict, Visualize)**

## 📂 Project Structure
  House_Price_Prediction/ │── README.md # Project Documentation │── requirements.txt # Dependencies List │── house_price_prediction.ipynb # Jupyter Notebook Version │ ├── data/ # Dataset (User Must Generate) │ ├── california_housing.csv │ ├── models/ # Folder for Trained Model (Not Uploaded) │ ├── random_forest_model.pkl # User Must Train │ ├── src/ # Python Scripts │ ├── train_model.py # Model Training Script │ ├── predict.py # Prediction Script │ ├── visuals/ # Graphs & Visualizations │ ├── feature_importance.png

## 📊 Dataset
- The dataset used is **California Housing Data** from Scikit-learn.
- Users must generate the dataset by running:
  ```python
  from sklearn.datasets import fetch_california_housing
  import pandas as pd

  data = fetch_california_housing()
  df = pd.DataFrame(data.data, columns=data.feature_names)
  df["Price"] = data.target
  df.to_csv("data/california_housing.csv", index=False)

  print("Dataset saved as data/california_housing.csv")
🚀 Installation & Usage
🔹 1️⃣ Clone the Repository
  
    git clone https://github.com/yourusername/House_Price_Prediction.git
    cd House_Price_Prediction
🔹 2️⃣ Install Dependencies
  
    pip install -r requirements.txt

🔹 3️⃣ Generate Dataset
    
    python src/generate_data.py
🔹 4️⃣ Train the Model
Since the trained random_forest_model.pkl is too large to be uploaded, users must train the model locally:

    python src/train_model.py

This will generate the missing model file inside the models/ folder.

🔹 5️⃣ Make Predictions
After training, users can make predictions:

    python src/predict.py

## 🔥 Results & Performance
  - R² Score: 0.8
  - Feature Importance: Visualized in visuals/feature_importance.png.
## ✨ Technologies Used
- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Joblib 
