import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.genfromtxt('house_prices.csv', delimiter=',', skip_header=1)
x_train = data[:, 0]  # Size in square meters
y_train = data[:, 1]  # Price in euros

# Normalize data for better performance of gradient descent
x_train_normalized = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_normalized = (y_train - np.mean(y_train)) / np.std(y_train)

# Data Visualization
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Prices')
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
y_pred = predict(x_train_normalized, w, b)

# Graph Plotting for Model Representation
plt.scatter(x_train_normalized, y_train_normalized, marker='x', c='r', label='Normalized Actual Prices')
plt.plot(x_train_normalized, y_pred, label='Model Prediction', color='blue')
plt.title("Normalized House Sizes vs. Prices Prediction")
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

# Compute and plot the cost function with respect to w
w_values = np.linspace(-2, 4, 50)
cost_values = [compute_cost(x_train_normalized, y_train_normalized, w, b) for w in w_values]

plt.figure()
plt.plot(w_values, cost_values, label='Cost Function Curve')
plt.title('Cost vs. Weight')
plt.xlabel('Weight (w)')
plt.ylabel('Cost (J)')
plt.legend()
plt.show()

# Compute gradient of the cost function
def compute_gradient(x, y, w, b):
    """Computes the gradient of the cost function with respect to parameters w and b."""
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Perform gradient descent to learn parameters
def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    """Performs gradient descent to update w and b."""
    w = w_init
    b = b_init
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % 100 == 0:
            J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

# Set hyperparameters and initialize parameters
alpha = 0.01
num_iters = 1000
w_init = 0
b_init = 0

# Run gradient descent
w_final, b_final, J_history = gradient_descent(x_train_normalized, y_train_normalized, w_init, b_init, alpha, num_iters)

print(f"Final parameters: w = {w_final}, b = {b_final}")

# Plot cost over iterations
plt.figure()
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function During Gradient Descent')
plt.show()

# Final Model Evaluation
# Reversing normalization to plot predictions vs. actual prices
x_train_pred = np.linspace(x_train.min(), x_train.max(), 100)
y_train_pred = predict((x_train_pred - np.mean(x_train)) / np.std(x_train), w_final, b_final) * np.std(y_train) + np.mean(y_train)

plt.figure()
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Prices')
plt.plot(x_train_pred, y_train_pred, label='Model Prediction', color='blue')
plt.title("House Sizes vs. Prices Prediction")
plt.xlabel("Size (square meters)")
plt.ylabel("Price (euros)")
plt.legend()
plt.show()
