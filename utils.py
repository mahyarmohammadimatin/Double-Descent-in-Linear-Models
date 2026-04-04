import numpy as np

def fit_least_squares(X, y): # TODO: change to self implemented function
    return np.linalg.pinv(X) @ y
def fit_ridge_regression(X, y, lam): # TODO: change to self implemented function
    n, d = X.shape
    I = np.eye(d)
    w_hat = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return w_hat
def predict(X, w):
    return X @ w
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)