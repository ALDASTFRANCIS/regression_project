from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

import os

file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'house_prices.csv')
df = pd.read_csv(os.path.abspath(file_path))
X = df[['area']]  # 2D
y = df['price']

model = LinearRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_}, Coefficient: {model.coef_}")

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.title("Linear Regression - sklearn")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()