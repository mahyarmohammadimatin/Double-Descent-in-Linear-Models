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
def get_metrics(y_true, y_pred):
    bias = np.mean(y_pred - y_true)
    variance = np.var(y_pred - y_true)
    MSE = mse(y_true, y_pred)
    bias_squared = bias ** 2
    bias_squared + variance
    return {'bias':bias, 'variance':variance, 'mse':MSE}

def merge_dictionaries(list_of_dicts):
    final_dict = {}
    for key in list_of_dicts[0]:
        final_dict[key] = []
        for dictionary in list_of_dicts:
            final_dict[key].append(dictionary[key])
    return final_dict
def avg_of_dictionaries(list_of_dicts):
    return {key: np.mean(value) for key,value in merge_dictionaries(list_of_dicts).items()}