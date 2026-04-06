import numpy as np

# Compute inverse using Gaussian elimination
def matrix_inverse(A):
    n = len(A)
    aug = np.zeros((n, 2 * n))  # Create augmented matrix [A | I]

    for i in range(n):
        for j in range(n):
            aug[i][j] = A[i][j]
        aug[i][n + i] = 1.0

    # Forward elimination
    for i in range(n):
        # Make diagonal = 1
        pivot = aug[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        for j in range(2 * n):
            aug[i][j] /= pivot

        for k in range(n): # Eliminate other rows
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]

    # Extract inverse from augmented matrix
    inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inv[i][j] = aug[i][j + n]

    return inv

def fit_least_squares(X, y, fast=False):
    if fast:
        return np.linalg.pinv(X) @ y
    # w_hat = (X^T X)^(-1) X^T y
    Xt = X.T
    XtX = np.dot(Xt, X)
    XtX_inv = matrix_inverse(XtX)
    Xty = np.dot(Xt, y) # Compute X^T y

    w_hat = np.dot(XtX_inv, Xty) # Final weights
    return w_hat
def fit_ridge_regression(X, y, lam, fast=False):
    n, d = X.shape
    I = np.eye(d)
    if fast:
        w_hat = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
        return w_hat
    # w_hat = (X^T X + lambda * I)^(-1) X^T y
    Xt = X.T
    XtX = np.dot(Xt, X)
    XtX_lam = XtX + lam * I  # Add regularization term
    XtX_lam_inv = matrix_inverse(XtX_lam)
    Xty = np.dot(Xt, y)

    w_hat = np.dot(XtX_lam_inv, Xty) # Final weights
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
def get_metrics(y_true, y_pred):
    bias = np.mean(y_pred - y_true)
    variance = np.var(y_pred - y_true)
    MSE = mse(y_true, y_pred)
    bias_squared = bias ** 2
    bias_squared + variance
    return {'bias':bias, 'variance':variance, 'mse':MSE, 'bias2':bias_squared}

def merge_dictionaries(list_of_dicts):
    final_dict = {}
    for key in list_of_dicts[0]:
        final_dict[key] = []
        for dictionary in list_of_dicts:
            final_dict[key].append(dictionary[key])
    return final_dict
def avg_of_dictionaries(list_of_dicts):
    return {key: np.mean(value) for key,value in merge_dictionaries(list_of_dicts).items()}