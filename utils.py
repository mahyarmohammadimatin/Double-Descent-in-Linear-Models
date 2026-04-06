import numpy as np

def fit_least_squares(X, y): # TODO: change to self implemented function
    return np.linalg.pinv(X) @ y
def fit_ridge_regression(X, y, lam): # TODO: change to self implemented function
    n, d = X.shape
    I = np.eye(d)
    w_hat = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return w_hat
def fit_least_squares_gd(X, y, lr=0.01, n_iters=200):
    n, d = X.shape
    # Initialize weights at zero (IMPORTANT for minimum-norm solution)
    w = np.zeros(d)
    for _ in range(n_iters):
        grad = (2 / n) * X.T @ (X @ w - y)
        w = w - lr * grad
    return w
def predict(X, w):
    return X @ w
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)