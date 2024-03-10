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
    w = w_init
    b = b_init
    J_history = []
    p_history = []  # To store the history of parameters
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % 100 == 0:  # Compute cost and parameters every 100 iterations
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
    return w, b, J_history, p_history


# Set hyperparameters and initialize parameters
alpha = 0.01
num_iters = 1000
w_init = 0
b_init = 0

# Run gradient descent
w_final, b_final, J_history, p_history = gradient_descent(x_train_normalized, y_train_normalized, w_init, b_init, alpha, num_iters)

print(f"Final parameters: w = {w_final}, b = {b_final}")

# Plot cost over iterations
plt.figure()
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function During Gradient Descent')
plt.show()

# Prepare the meshgrid for contour plot based on w and b ranges
w_range = np.linspace(-2, 4, 100)
b_range = np.linspace(-5, 5, 100)
W, B = np.meshgrid(w_range, b_range)

# Compute cost for each combination of w and b
Z = np.array([
    compute_cost(x_train_normalized, y_train_normalized, w, b) 
    for w, b in zip(np.ravel(W), np.ravel(B))
]).reshape(W.shape)

# Contour plot
plt.figure(figsize=(12, 6))
cp = plt.contour(W, B, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.colorbar(cp)
plt.title('Cost Function Contour with Gradient Descent Path')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')

# Extract w and b values from p_history
w_values, b_values = zip(*p_history)
# Overlay the gradient descent path
plt.plot(w_values, b_values, 'r--', marker='x', label='Gradient Descent Path')
plt.legend()
plt.show()

# High learning rate
high_alpha = 0.5
high_iter = 1000

# Run gradient descent with a high learning rate
w_high, b_high, J_history_high, p_history_high = gradient_descent(x_train_normalized, y_train_normalized, w_init, b_init, high_alpha, high_iter)

# Plotting the cost function over iterations to show divergence
plt.figure()
plt.plot(J_history_high, label=f'Learning Rate = {high_alpha}')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Divergence with High Learning Rate')
plt.legend()
plt.show()

# Generate mesh grid for contour plot
W, B = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))  # Example range adjustments
Z = np.array([
    compute_cost(x_train_normalized, y_train_normalized, w, b) 
    for w, b in zip(np.ravel(W), np.ravel(B))
])
Z = Z.reshape(W.shape)


# Contour plot for Cost with respect to w and b showing the divergence
plt.figure(figsize=(12, 6))
cp = plt.contour(W, B, Z, levels=np.logspace(-2, 5, 50), cmap='viridis')
plt.colorbar(cp)
plt.title('Cost Function Contour with Divergent Gradient Descent Path')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')

# Extract w and b values from p_history_high for high learning rate
w_values_high, b_values_high = zip(*p_history_high)
plt.plot(w_values_high, b_values_high, 'r--', marker='x', label='Divergent Path (High Alpha)')
plt.legend()
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
