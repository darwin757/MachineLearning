import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('regression\house_prices_multivariate.csv')

# Separate the features and the target variable
X = df.drop('Price', axis=1).values
y = df['Price'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Features for later use in plotting
X_features = df.drop('Price', axis=1).columns.tolist()

# Scale/normalize the training data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)  # Important: Use the same scaling for the test set

# Create and fit the regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_train_norm, y_train)

# Make predictions on the test set
y_pred_test = sgdr.predict(X_test_norm)

# Plotting for all features
num_features = len(X_features)
fig, axs = plt.subplots(1, num_features, figsize=(20, 5), sharey=True)
for i in range(num_features):
    axs[i].scatter(X_test[:, i], y_test, label='Actual Price', color='blue')
    axs[i].scatter(X_test[:, i], y_pred_test, label='Predicted Price', color='orange', alpha=0.7)
    axs[i].set_title(f'{X_features[i]} vs. Price')
    axs[i].set_xlabel(X_features[i])
axs[0].set_ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Squared Error on Test Set: {mse}")
print(f"R^2 Score on Test Set: {r2}")
