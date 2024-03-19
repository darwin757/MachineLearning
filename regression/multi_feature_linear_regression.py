import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('regression\house_prices_multivariate.csv')

# Separate features and target variable
X = df.iloc[:, :-1].values  # All rows, all columns except the last
y = df.iloc[:, -1].values   # All rows, last column

# Normalize features for better performance of gradient descent
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
y_normalized = (y - y.mean()) / y.std()

# Add intercept term to X_normalized
X_normalized = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])

# Splitting the data into training and testing datasets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(len(X_normalized) * split_ratio)
X_train, X_test = X_normalized[:split_index], X_normalized[split_index:]
y_train, y_test = y_normalized[:split_index], y_normalized[split_index:]

# Gradient descent function for multivariate linear regression
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        delta = (1 / m) * (X.T.dot(errors))
        theta -= alpha * delta
        J_history.append((1 / (2 * m)) * np.dot(errors.T, errors))
    return theta, J_history

# Initialize fitting parameters
theta = np.zeros(X_train.shape[1])

# Some gradient descent settings
iterations = 1000
alpha = 0.01

# Run gradient descent
theta, J_history = gradient_descent(X_train, y_train, theta, alpha, iterations)

# Plot the convergence graph
plt.figure()
plt.plot(range(len(J_history)), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Cost Function Decrease Over Iterations')
plt.show()

# Predict function
def predict(X, theta):
    return X.dot(theta)

# Predict values for training set
y_train_pred = predict(X_train, theta)

# Predict values for testing set
y_test_pred = predict(X_test, theta)

# Function to compute the coefficient of determination, also known as R squared
def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Function to compute Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Function to compute Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to compute Root Mean Squared Error
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Performance evaluation
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r_squared(y_test, y_test_pred)

# Adjusted R-squared calculation
n = len(y_test)  # Number of observations
k = X_test.shape[1] - 1  # Number of features
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Creating a DataFrame to display the metrics
metrics_df = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared', 'Adjusted R-squared'],
    'Value': [mae, mse, rmse, r2, adj_r2]
})

# Print the metrics
print(metrics_df.to_string(index=False))

# Print out the parameters for the model
print('Model parameters:')
params = ['Intercept'] + df.columns[:-1].tolist()
for param, value in zip(params, theta):
    print(f'{param}: {value:.4f}')
