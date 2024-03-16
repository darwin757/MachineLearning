import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_polynomial_features(X, degree=2):
    """
    Manually add polynomial features up to a specified degree.
    """
    X_poly = X.copy()
    for d in range(2, degree+1):
        for i in range(X.shape[1]):
            X_poly = np.hstack((X_poly, (X[:, i]**d).reshape(-1, 1)))
    return X_poly

def feature_normalize(X):
    """
    Normalize features for better performance of gradient descent.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_normalized = (X - mu) / sigma
    return X_normalized, mu, sigma

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Gradient descent function for multivariate regression.
    """
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        delta = (1 / m) * (X.T.dot(errors))
        theta -= alpha * delta
        cost = (1 / (2 * m)) * np.dot(errors.T, errors)
        J_history.append(cost)
        if i % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {i}, Cost: {cost}")
    return theta, J_history

def predict(X, theta):
    """
    Predict function.
    """
    return X.dot(theta)

def r_squared(y, y_pred):
    """
    Calculate the R-squared value to evaluate the model.
    """
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_metrics(y_true, y_pred, X):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    n = len(y_true)
    k = X.shape[1] - 1  # Adjusting for intercept term
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return mae, mse, rmse, r2, adj_r2

# Load data from CSV file
df = pd.read_csv('house_prices_multivariate.csv')

# Separate features and target variable
X = df.iloc[:, :-1].values  # All rows, all columns except the last
y = df.iloc[:, -1].values   # All rows, last column

# Create polynomial features
degree = 2
X_poly = add_polynomial_features(X, degree=degree)

# Normalize polynomial features
X_poly_normalized, _, _ = feature_normalize(X_poly)

# Add intercept term to normalized polynomial features
X_poly_normalized = np.hstack([np.ones((X_poly_normalized.shape[0], 1)), X_poly_normalized])

# Initialize fitting parameters for polynomial model
theta_poly = np.zeros(X_poly_normalized.shape[1])

# Some gradient descent settings
iterations = 1000
alpha = 0.01

# Run gradient descent for polynomial regression
theta_poly, J_history_poly = gradient_descent(X_poly_normalized, y, theta_poly, alpha, iterations)

# Predict values for polynomial regression
y_pred_poly = predict(X_poly_normalized, theta_poly)

# Compute and print metrics for Polynomial Regression
mae_poly, mse_poly, rmse_poly, r2_poly, adj_r2_poly = compute_metrics(y, y_pred_poly, X_poly_normalized)
print(f"Polynomial Regression Metrics:\n MAE: {mae_poly:.4f}\n MSE: {mse_poly:.4f}\n RMSE: {rmse_poly:.4f}\n R-squared: {r2_poly:.4f}\n Adjusted R-squared: {adj_r2_poly:.4f}\n")

# Visualizing cost decrease
plt.figure(figsize=(12, 6))
plt.plot(range(iterations), J_history_poly, label='Polynomial Regression', color='orange')
plt.title(f'Polynomial Regression (degree {degree}) Training Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.legend()
plt.show()

# Plotting Actual vs. Predicted Prices for polynomial regression
plt.figure(figsize=(14, 7))
plt.scatter(y, y, alpha=0.5, label='Actual', color='grey')
plt.scatter(y, y_pred_poly, alpha=0.5, color='orange', label='Predicted')
plt.title('Polynomial Regression\nActual vs. Predicted Prices on Normalized Data')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.tight_layout()
plt.show()

# Plot predictions against each feature
feature_names = ['Size', 'Bedrooms', 'Age', 'Bathrooms', 'DistanceToCityCenter', 'SizePerBedroom']
for i in range(X.shape[1]):
    plt.figure(figsize=(14, 7))
    plt.scatter(X[:, i], y, alpha=0.8, label='Actual Prices', color='grey')
    plt.scatter(X[:, i], y_pred_poly, alpha=0.6, label='Predicted Prices', color='orange')
    plt.title(f'Polynomial Regression\n{feature_names[i]} vs. Price')
    plt.xlabel(feature_names[i])
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
