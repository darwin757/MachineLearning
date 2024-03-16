import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.genfromtxt('house_prices.csv', delimiter=',', skip_header=1)
x = data[:, 0]  # Size in square meters
y = data[:, 1]  # Price in euros

# Manually splitting the dataset into 80% training and 20% testing
np.random.seed(42)  # For reproducibility
shuffled_indices = np.random.permutation(len(x))
train_set_size = int(len(x) * 0.8)
train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]

# Normalize data for better performance of gradient descent
x_train_normalized = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_normalized = (y_train - np.mean(y_train)) / np.std(y_train)

x_test_normalized = (x_test - np.mean(x_train)) / np.std(x_train)  # Use training mean and std
y_test_normalized = (y_test - np.mean(y_train)) / np.std(y_train)  # Use training mean and std

# Data Visualization
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Prices (Train)')
plt.scatter(x_test, y_test, marker='o', c='b', label='Actual Prices (Test)', alpha=0.5)
plt.title("House Sizes vs. Prices")
plt.xlabel("Size (square meters)")
plt.ylabel("Price (euros)")
plt.legend()
plt.show()

# Model parameters (initial guess)
w = np.random.rand()
b = np.random.rand()

# Predict the price of a house given its size
def predict(x, w, b):
    """Returns the model prediction for a given house size."""
    return w * x + b

# Model prediction on normalized data
y_pred_train = predict(x_train_normalized, w, b)

# Graph Plotting for Model Representation (Train Data)
plt.scatter(x_train_normalized, y_train_normalized, marker='x', c='r', label='Normalized Actual Prices (Train)')
plt.plot(x_train_normalized, y_pred_train, label='Model Prediction', color='blue')
plt.title("Normalized House Sizes vs. Prices Prediction (Train)")
plt.xlabel("Normalized Size")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()

# Compute cost for linear regression
def compute_cost(x, y, w, b):
    """Computes the cost function for given features, targets, and parameters."""
    m = x.shape[0]
    total_cost = (1 / (2 * m)) * np.sum((predict(x, w, b) - y) ** 2)
    return total_cost

# Compute gradient of the cost function
def compute_gradient(x, y, w, b):
    """Computes the gradient of the cost function with respect to parameters w and b."""
    m = x.shape[0]
    dj_dw = (1 / m) * np.sum((predict(x, w, b) - y) * x)
    dj_db = (1 / m) * np.sum(predict(x, w, b) - y)
    return dj_dw, dj_db

# Perform gradient descent to learn parameters
def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

# Set hyperparameters and initialize parameters
alpha = 0.01
num_iters = 1000
w_init = 0
b_init = 0

# Run gradient descent (Train Data)
w_final, b_final, J_history = gradient_descent(x_train_normalized, y_train_normalized, w_init, b_init, alpha, num_iters)

print(f"Final parameters (Train): w = {w_final}, b = {b_final}")

# Plot cost over iterations (Train Data)
plt.figure()
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function During Gradient Descent (Train)')
plt.show()

# Prediction and Evaluation on Test Data
y_pred_test = predict(x_test_normalized, w_final, b_final)
plt.scatter(x_test, y_test, marker='x', c='r', label='Actual Prices (Test)')
plt.scatter(x_test, (y_pred_test * np.std(y_train)) + np.mean(y_train), label='Predicted Prices (Test)', color='blue')
plt.title("House Sizes vs. Prices Prediction (Test)")
plt.xlabel("Size (square meters)")
plt.ylabel("Price (euros)")
plt.legend()
plt.show()
