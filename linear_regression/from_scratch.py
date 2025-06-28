import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'house_prices.csv')
df = pd.read_csv(os.path.abspath(file_path))
X = df['area'].values
y = df['price'].values

# Mean normalization
X_mean = np.mean(X)
y_mean = np.mean(y)

# Calculate slope (β1) and intercept (β0)
numerator = sum((X - X_mean) * (y - y_mean))
denominator = sum((X - X_mean)**2)
b1 = numerator / denominator
b0 = y_mean - b1 * X_mean

print(f"Model: y = {b0:.2f} + {b1:.2f}x")

# Predict and plot
y_pred = b0 + b1 * X
plt.scatter(X, y, label='True data')
plt.plot(X, y_pred, color='red', label='Predicted line')
plt.legend()
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression - From Scratch")
plt.show()