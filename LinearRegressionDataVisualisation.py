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

# Model Representation
def predict(x, w, b):
    return w * x + b

# Model prediction on normalized data
y_pred = predict(x_train_normalized, w, b)

# Graph Plotting for Model Representation
# Plotting the model prediction with initial random parameters
plt.scatter(x_train_normalized, y_train_normalized, marker='x', c='r', label='Normalized Actual Prices')
plt.plot(x_train_normalized, y_pred, label='Model Prediction', color='blue')
plt.title("Normalized House Sizes vs. Prices Prediction")
plt.xlabel("Normalized Size")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()

def compute_cost(x, y, w, b): 
    m = x.shape[0]
    total_cost = (1 / (2 * m)) * np.sum((predict(x, w, b) - y) ** 2)
    return total_cost

# Example usage of compute_cost
w_example = 0.1
b_example = 0.0
cost_example = compute_cost(x_train_normalized, y_train_normalized, w_example, b_example)
print(f"Example Cost with w = {w_example} and b = {b_example}: {cost_example}")

# Visualizing the Cost Function with respect to w
w_values = np.linspace(-2, 4, 50)  # Adjust range and density as needed
cost_values = [compute_cost(x_train_normalized, y_train_normalized, w, b_example) for w in w_values]

plt.figure()
plt.plot(w_values, cost_values, label='Cost Function Curve')
plt.title('Cost vs. Weight')
plt.xlabel('Weight (w)')
plt.ylabel('Cost (J)')
plt.legend()
plt.show()

# Contour plot for Cost with respect to w and b
W, B = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-5, 5, 100))
Z = np.array([compute_cost(x_train_normalized, y_train_normalized, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
Z = Z.reshape(W.shape)

plt.figure()
cp = plt.contour(W, B, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.colorbar(cp)
plt.title('Cost Function Contour')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.show()

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % 100 == 0:  # Compute cost every 100 iterations
            J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

# Set hyperparameters
alpha = 0.01
num_iters = 1000

# Initialize parameters
w_init = 0
b_init = 0

# Run gradient descent
w_final, b_final, J_history = gradient_descent(x_train_normalized, y_train_normalized, w_init, b_init, alpha, num_iters)

print(f"Final parameters: w = {w_final}, b = {b_final}")

# Plot the cost over iterations
plt.figure()
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function During Gradient Descent')
plt.show()

# Generate mesh grid for contour plot
W, B = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-5, 5, 100))
Z = np.array([compute_cost(x_train_normalized, y_train_normalized, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
Z = Z.reshape(W.shape)

# Contour plot for Cost with respect to w and b showing the optimization path
plt.figure()
cp = plt.contour(W, B, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.plot(w_final, b_final, 'r*', markersize=10)  # Mark the final parameters
plt.colorbar(cp)
plt.title('Cost Function Contour with Optimization Path')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.show()
