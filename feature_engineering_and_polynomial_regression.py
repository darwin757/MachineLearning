import numpy as np
import matplotlib.pyplot as plt

def zscore_normalize_features(X):
    """Normalize features using Z-score normalization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

def run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2):
    """Run gradient descent algorithm."""
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # add bias term
    theta = np.zeros(n + 1)
    cost_history = []
    
    for _ in range(iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= alpha * gradients
        cost = np.mean((X_b.dot(theta) - y) ** 2)
        cost_history.append(cost)
    
    return theta[1:], theta[0]  # Return model_w (weights) and model_b (bias)

np.set_printoptions(precision=2)

# Data generation and polynomial regression fitting with various features
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.plot(x, X.dot(model_w) + model_b, label="Predicted Value")
plt.title("No Feature Engineering")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Adding polynomial features x**2
X = x**2
X = X.reshape(-1, 1)
model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-5)
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.plot(x, X.dot(model_w) + model_b, label="Predicted Value")
plt.title("Added x**2 Feature")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Adding multiple polynomial features: x, x**2, x**3
X = np.c_[x, x**2, x**3]
model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.plot(x, X.dot(model_w) + model_b, label="Predicted Value")
plt.title("x, x**2, x**3 Features")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Feature normalization and fitting a complex function
x = np.arange(0, 20, 1)
y = np.cos(x/2)
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X)
model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha=1e-1)
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.plot(x, X.dot(model_w) + model_b, label="Predicted Value")
plt.title("Normalized Features for Complex Function Fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
