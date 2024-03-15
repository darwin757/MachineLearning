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
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        delta = (1 / m) * (X.T.dot(errors))
        theta -= alpha * delta
        J_history.append((1 / (2 * m)) * np.dot(errors.T, errors))
    return theta, J_history

def predict(X, theta):
    """
    Predict function.
    """
    return X.dot(theta)

# Evaluation metrics functions
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


# Normalize original features
X_normalized, _, _ = feature_normalize(X)

# Add intercept term to normalized features
X_normalized = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])

# Create polynomial features
degree = 2
X_poly = add_polynomial_features(X, degree=degree)

# Normalize polynomial features
X_poly_normalized, _, _ = feature_normalize(X_poly)

# Add intercept term to normalized polynomial features
X_poly_normalized = np.hstack([np.ones((X_poly_normalized.shape[0], 1)), X_poly_normalized])

# Initialize fitting parameters for both models
theta_linear = np.zeros(X_normalized.shape[1])
theta_poly = np.zeros(X_poly_normalized.shape[1])

# Some gradient descent settings
iterations = 1000
alpha = 0.01

# Initialize fitting parameters for both models on original data
theta_linear_original = np.zeros(X.shape[1] + 1)
theta_poly_original = np.zeros(X_poly.shape[1] + 1)

# Add intercept term to original features and polynomial features
X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
X_poly_with_intercept = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])

# Run gradient descent for linear regression on original data
theta_linear_original, J_history_linear_original = gradient_descent(
    X_with_intercept, y, theta_linear_original, alpha, iterations
)

# Run gradient descent for polynomial regression on original data
theta_poly_original, J_history_poly_original = gradient_descent(
    X_poly_with_intercept, y, theta_poly_original, alpha, iterations
)

# Predict values for linear regression on original data
y_pred_linear_original = predict(X_with_intercept, theta_linear_original)

# Predict values for polynomial regression on original data
y_pred_poly_original = predict(X_poly_with_intercept, theta_poly_original)

# Run gradient descent for linear regression
theta_linear, J_history_linear = gradient_descent(X_normalized, y, theta_linear, alpha, iterations)

# Run gradient descent for polynomial regression
theta_poly, J_history_poly = gradient_descent(X_poly_normalized, y, theta_poly, alpha, iterations)

# Predict values for linear regression
y_pred_linear = predict(X_normalized, theta_linear)

# Predict values for polynomial regression
y_pred_poly = predict(X_poly_normalized, theta_poly)

# Compute R-squared for both models
def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

r2_linear = r_squared(y, y_pred_linear)
r2_poly = r_squared(y, y_pred_poly)

# Compute and print metrics for Linear Regression
mae_linear, mse_linear, rmse_linear, r2_linear, adj_r2_linear = compute_metrics(y, y_pred_linear, X_normalized)
print(f"Linear Regression Metrics:\n MAE: {mae_linear:.4f}\n MSE: {mse_linear:.4f}\n RMSE: {rmse_linear:.4f}\n R-squared: {r2_linear:.4f}\n Adjusted R-squared: {adj_r2_linear:.4f}\n")

# Compute and print metrics for Polynomial Regression
mae_poly, mse_poly, rmse_poly, r2_poly, adj_r2_poly = compute_metrics(y, y_pred_poly, X_poly_normalized)
print(f"Polynomial Regression Metrics:\n MAE: {mae_poly:.4f}\n MSE: {mse_poly:.4f}\n RMSE: {rmse_poly:.4f}\n R-squared: {r2_poly:.4f}\n Adjusted R-squared: {adj_r2_poly:.4f}\n")


# Visualizing cost decrease
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(iterations), J_history_linear, label='Linear Regression')
plt.title('Linear Regression Training Convergence')

plt.subplot(1, 2, 2)
plt.plot(range(iterations), J_history_poly, label='Polynomial Regression', color='orange')
plt.title(f'Polynomial Regression (degree {degree}) Training Convergence')

plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.legend()
plt.show()

# Plotting Actual vs. Predicted Prices for both models
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(y, y_pred_linear, alpha=0.5, label='Predicted')
plt.scatter(y, y,alpha=0.5, label='Actual', color='grey')
plt.title('Linear Regression\nActual vs. Predicted Prices on Normalized Data')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y, y_pred_poly, alpha=0.5, color='orange', label='Predicted')
plt.scatter(y, y, alpha=0.5, label='Actual', color='grey')
plt.title('Polynomial Regression\nActual vs. Predicted Prices Prices on Normalized Data')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting Actual vs. Predicted Prices for both models as scatter plots
plt.figure(figsize=(14, 7))

# Linear Regression on original data
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred_linear, alpha=0.5, label='Predicted Linear')
plt.scatter(y, y, alpha=0.5, label='Actual', color='grey')
plt.title('Linear Regression on Original Data\nActual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

# Polynomial Regression on original data
plt.subplot(1, 2, 2)
plt.scatter(y, y_pred_poly, alpha=0.5, color='orange', label='Predicted Poly')
plt.scatter(y, y, alpha=0.5, label='Actual', color='grey')
plt.title('Polynomial Regression on Original Data\nActual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

plt.tight_layout()
plt.show()

