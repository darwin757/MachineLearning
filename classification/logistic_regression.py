import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from scipy.special import expit as sigmoid

# Load the data
data = pd.read_csv('tumor_classification_dataset.csv')
X = data.drop('Target', axis=1).values
y = data['Target'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the logistic regression model
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_

    def initialize_parameters(self, n):
        self.w = np.zeros((n, 1))
        self.b = 0
    
    def compute_cost(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)  # Ensure y is a column vector
        cost = (-1 / m) * np.sum(y * np.log(sigmoid(np.dot(X, self.w) + self.b)) + (1 - y) * np.log(1 - sigmoid(np.dot(X, self.w) + self.b)))
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.w))
        total_cost = cost + reg_cost
        return total_cost
    
    def gradient_descent(self, X, y):
        m, n = X.shape
        cost_history = []
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        for i in range(self.num_iterations):
            dz = sigmoid(np.dot(X, self.w) + self.b) - y
            dw = (1 / m) * np.dot(X.T, dz) + (self.lambda_ / m) * self.w
            db = (1 / m) * np.sum(dz)
        
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                cost_history.append(cost)
                print(f"Iteration {i}: Cost {cost}")  # Optional: Print cost every 100 iterations

        return self.w, self.b, cost_history

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.w) + self.b) >= 0.5
        return y_pred.astype(int)
    
    def fit(self, X, y):
        self.initialize_parameters(X.shape[1])
        self.w, self.b, self.cost_history = self.gradient_descent(X, y)

# Train the custom logistic regression model
model = LogisticRegressionCustom(learning_rate=0.1, num_iterations=1000, lambda_=0.1)
model.fit(X_train_scaled, y_train)
# Evaluate the model
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
