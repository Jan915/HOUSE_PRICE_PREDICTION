# HOUSE_PRICE_PREDICTION
This project aims to predict house prices using a Kaggle dataset. It includes data cleaning, exploratory data analysis (EDA), feature selection, and a simple Linear Regression model for prediction. The project demonstrates how data analytics and basic machine learning techniques can be applied to solve real-world problems like price prediction.  


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
# Replace 'house_prices.csv' with your dataset's filename
df = pd.read_csv('house_prices.csv')

# Display basic information
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Exploratory Data Analysis (EDA)
print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Selecting features and target variable
# Update column names based on your dataset
X = df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Visualizing predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
