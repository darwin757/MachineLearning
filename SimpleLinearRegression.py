import numpy as np
import pandas as pd  # Import pandas for table formatting

# Load data from CSV file
data = np.genfromtxt('house_prices.csv', delimiter=',', skip_header=1)
X = data[:, 0]  # Size in square meters
y = data[:, 1]  # Price in euros

# Normalize data for better performance of gradient descent
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)

# Splitting the data into training and testing datasets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initializing parameters
theta0 = 0
theta1 = 0
learning_rate = 0.01  # Adjusted for normalized data
iterations = 1000

# Gradient descent function
def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        prediction = theta0 + theta1 * X
        theta0 -= learning_rate * (1/m) * sum(prediction - y)
        theta1 -= learning_rate * (1/m) * sum((prediction - y) * X)
    return theta0, theta1

# Training the model
theta0, theta1 = gradient_descent(X_train, y_train, theta0, theta1, learning_rate, iterations)

# Predict function
def predict(X):
    return theta0 + theta1 * X

# Prediction on testing data
predictions = predict(X_test)

# Performance metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_tot = sum((y_true - mean_y) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Calculating and displaying metrics
mse = mean_squared_error(y_test, predictions)
r2 = r_squared(y_test, predictions)

# Creating DataFrame for neat display
results_df = pd.DataFrame({
    'Test Size (Normalized)': X_test,
    'Actual Price (Normalized)': y_test,
    'Predicted Price (Normalized)': predictions
})

print(results_df)  # Print the DataFrame
print(f"\nMean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
