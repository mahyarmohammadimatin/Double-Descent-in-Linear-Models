import numpy as np

# ---------------------------------------- Machine Learning Models From Scratch ----------------------------------------
def fit_least_squares(X, y, fast=False):
    if fast:
        return np.linalg.pinv(X) @ y
    # w_hat = (X^T X)^(-1) X^T y
    Xt = X.T
    XtX = np.dot(Xt, X)
    # Check if singular (rank deficient)
    if matrix_rank(XtX) < XtX.shape[0]:
        X_pinv = pseudoinverse_svd(X)  # Use pseudoinverse
        w_hat = np.dot(X_pinv, y)
    else: # Use normal equation
        XtX_inv = matrix_inverse(XtX)
        Xty = np.dot(Xt, y)
        w_hat = np.dot(XtX_inv, Xty)
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

# ---------------------------------------- Linear Algebra from Scratch ----------------------------------------

# Performs Gauss-Jordan elimination.
def row_reduce(A, augment=None):
    A = A.astype(float).copy()
    n, m = A.shape

    if augment is not None:
        augment = augment.astype(float).copy()

    row = 0
    for col in range(m):
        if row >= n:
            break

        # Find pivot
        pivot = None
        for r in range(row, n):
            if abs(A[r, col]) > 1e-10:
                pivot = r
                break

        if pivot is None:
            continue

        # Swap rows
        A[[row, pivot]] = A[[pivot, row]]
        if augment is not None:
            augment[[row, pivot]] = augment[[pivot, row]]

        # Normalize pivot row
        pivot_val = A[row, col]
        A[row] /= pivot_val
        if augment is not None:
            augment[row] /= pivot_val

        for r in range(n):  # Eliminate other rows
            if r != row:
                factor = A[r, col]
                A[r] -= factor * A[row]
                if augment is not None:
                    augment[r] -= factor * augment[row]

        row += 1

    return (A, augment) if augment is not None else A


def matrix_rank(A, fast=True):
    if fast:
        return np.linalg.matrix_rank(A)
    R = row_reduce(A)
    rank = 0
    for row in R:
        if np.any(np.abs(row) > 1e-10):
            rank += 1
    return rank

def matrix_inverse(A, fast=True):
    if fast:
        return np.linalg.inv(A)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")
    I = np.eye(n)
    R, I_transformed = row_reduce(A, augment=I)
    # Check if left side became identity
    if not np.allclose(R, np.eye(n), atol=1e-6):
        raise ValueError("Matrix is singular and not invertible")
    return I_transformed

def pseudoinverse_svd(X):
    U, S, V = svd(X)
    S_inv = np.zeros((V.shape[1], U.shape[1])) # Build Sigma^+

    for i in range(len(S)):
        if S[i] > 1e-10:
            S_inv[i, i] = 1.0 / S[i]

    return np.dot(V, np.dot(S_inv, U.T)) # X^+ = V S^+ U^T

def svd(X, fast=True):
    if fast:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt.T
        return U, s, V
    XtX = np.dot(X.T, X)
    eigenvalues, V = eigenvalue(XtX)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    # Compute U
    U = []
    for i in range(len(singular_values)):
        sigma = singular_values[i]
        if sigma > 1e-10:
            u = np.dot(X, V[:, i]) / sigma
        else:
            u = np.zeros(X.shape[0])
        U.append(u)

    U = np.column_stack(U)

    return U, singular_values, V

def eigenvalue(XtX, fast=True, max_iter=1000, tol=1e-10):
    if fast:
        return np.linalg.eig(XtX)
    n = XtX.shape[0]
    V = np.eye(n)
    A = XtX.copy().astype(float)

    for _ in range(max_iter):
        # QR decomposition
        Q, R = qr_decomposition(A)
        A_new = R @ Q
        V = V @ Q

        # Check convergence (off-diagonal elements become small)
        off_diag_sum = np.sum(np.abs(np.tril(A_new, -1))) + np.sum(np.abs(np.triu(A_new, 1)))
        if off_diag_sum < tol:
            break
        A = A_new

    eigenvalues = np.diag(A) # Extract eigenvalues from diagonal

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    return eigenvalues, V

def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = vector_norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = np.zeros(m)
    return Q, R

def vector_norm(v):
    return np.sqrt(np.sum(v ** 2))

# ---------------------------------------- Python Utils ----------------------------------------
def merge_dictionaries(list_of_dicts):
    final_dict = {}
    for key in list_of_dicts[0]:
        final_dict[key] = []
        for dictionary in list_of_dicts:
            final_dict[key].append(dictionary[key])
    return final_dict
def avg_of_dictionaries(list_of_dicts):
    return {key: np.mean(value) for key,value in merge_dictionaries(list_of_dicts).items()}