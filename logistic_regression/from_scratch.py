import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# Load Data
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes.csv')
df = pd.read_csv(os.path.abspath(file_path))
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Add bias
X = np.hstack((np.ones((X.shape[0], 1)), X))

weights = np.zeros(X.shape[1])
lr = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    z = np.dot(X, weights)
    h = sigmoid(z)
    gradient = np.dot(X.T, (h - y)) / y.size
    weights -= lr * gradient

print(f"Final weights: {weights}")

# Accuracy
preds = predict(X, weights) >= 0.5
accuracy = (preds == y).mean()
print(f"Accuracy: {accuracy:.2f}")
