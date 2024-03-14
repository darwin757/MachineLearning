import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and display the dataset
data_path = 'house_prices_multivariate.csv'  # Update this path
df = pd.read_csv(data_path)
print(df.head())

# Separate features and target variable
X_df = df.drop('Price', axis=1)
y = df['Price'].values

# Feature normalization function
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    X_mean = X - mu
    return X_norm, mu, sigma, X_mean


X = df.iloc[:, :-1].values  # Exclude the last column (target variable)
y = df.iloc[:, -1].values.reshape(-1,1)  # Ensure y is a column vector
X = X_df.values
X_norm, mu, sigma, X_mean = feature_normalize(X)

# Adding intercept term
m = len(y)
X_norm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# Compute Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum(np.square(X.dot(theta) - y))
    return J

# Gradient Descent Function
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * X.T.dot(X.dot(theta) - y)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

# Initialize fitting parameters
theta = np.zeros((X_norm.shape[1], 1))
iterations = 1500
alpha = 0.01

# Run gradient descent
theta, J_history = gradient_descent(X_norm, y, theta, alpha, iterations)

# Plot the convergence graph
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence of Gradient Descent')
plt.show()


# Predict function
def predict(X, theta):
    return X.dot(theta)

# Generate predictions for the dataset
predictions = predict(X_norm, theta)

# Plotting
feature_names = X_df.columns
n_features = len(feature_names)

for i in range(n_features):
    for j in range(i+1, n_features):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # Unnormalized Data
        ax[0].scatter(X[:, i], X[:, j], alpha=0.5)
        ax[0].set_xlabel(feature_names[i])
        ax[0].set_ylabel(feature_names[j])
        ax[0].set_title("Unnormalized")
        ax[0].axis('equal')

        # Data after subtracting mean
        ax[1].scatter(X_mean[:, i], X_mean[:, j], alpha=0.5, color='orange')
        ax[1].set_xlabel(feature_names[i])
        ax[1].set_ylabel(feature_names[j])
        ax[1].set_title(r"$X - \mu$")
        ax[1].axis('equal')

        # Z-score Normalized Data
        ax[2].scatter(X_norm[:, i], X_norm[:, j], alpha=0.5, color='green')
        ax[2].set_xlabel(feature_names[i])
        ax[2].set_ylabel(feature_names[j])
        ax[2].set_title("Z-score normalized")
        ax[2].axis('equal')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"Distribution of {feature_names[i]} vs. {feature_names[j]} before, during, and after normalization")
        plt.show()

# Plot each feature vs. the Price
feature_names = df.columns[:-1]  # Exclude 'Price' from features
for feature_name in feature_names:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature_name], df['Price'], alpha=0.5)
    plt.title(f'{feature_name} vs. Price')
    plt.xlabel(feature_name)
    plt.ylabel('Price ($)')
    plt.show()

# Get the 'Price' column for plotting
price = df['Price'].values

# Feature names excluding 'Price'
feature_names = df.columns[:-1]  # Assuming the last column is 'Price'

# Plot each normalized feature against the Price
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_norm[:, i], price, alpha=0.5)
    plt.title(f'Normalized {feature_name} vs. Price')
    plt.xlabel(f'Normalized {feature_name}')
    plt.ylabel('Price ($)')
    plt.show()

# Histogram of feature distributions before and after normalization
features = df.columns[:-1]
fig, axs = plt.subplots(2, len(features), figsize=(20, 10))

for i, feature in enumerate(features):
    axs[0, i].hist(X[:, i], bins=20, alpha=0.5, color='b')
    axs[0, i].set_title(f'Original {feature}')

    axs[1, i].hist(X_norm[:, i+1], bins=20, alpha=0.5, color='g')  # i+1 to skip intercept term
    axs[1, i].set_title(f'Normalized {feature}')

plt.tight_layout()
plt.show()

# Assuming theta, X_norm, and df are already defined from previous steps
# Extract just the normalized features (excluding the intercept term) for plotting
X_norm_features = X_norm[:, 1:]

# Predict target using normalized features
predictions = predict(X_norm, theta).flatten()

# Plot predictions and targets versus original features
fig, ax = plt.subplots(1, len(feature_names), figsize=(20, 4), sharey=True)
for i, feature_name in enumerate(feature_names):
    ax[i].scatter(df[feature_name], y, label='Target', alpha=0.5, edgecolor='k')
    ax[i].scatter(df[feature_name], predictions, label='Prediction', alpha=0.5, color='orange', edgecolor='k')
    ax[i].set_xlabel(feature_name)
    if i == 0:
        ax[i].set_ylabel('Price ($)')
        ax[i].legend()
fig.suptitle("Target vs. Prediction Using Z-score Normalized Model for Each Feature")
plt.tight_layout()
plt.show()


# Plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(np.arange(len(J_history)), J_history)
ax2.plot(100 + np.arange(len(J_history[100:])), J_history[100:])
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('Iteration step')
ax2.set_xlabel('Iteration step')
plt.show()


# Visualize Actual vs Predicted Prices for the entire dataset
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()
