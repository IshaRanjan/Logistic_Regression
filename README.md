import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("car_price_data.csv")


X = data[['mileage', 'age']].values  # Features: mileage and age
y = data['price'].values  # Target variable: car price

X = np.c_[np.ones(X.shape[0]), X]


def compute_theta(X, y):
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    return theta

theta = compute_theta(X, y)

def predict(X, theta):
    return X.dot(theta)
y_pred = predict(X, theta)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r2 = r_squared(y, y_pred)
print(f"R-squared: {r2}")

plt.scatter(data['mileage'], y, color='blue', label='Actual prices')
plt.scatter(data['mileage'], y_pred, color='red', label='Predicted prices')
plt.xlabel('Mileage')
plt.ylabel('Car Price')
plt.legend()
plt.title('Car Price Prediction')
plt.show()
