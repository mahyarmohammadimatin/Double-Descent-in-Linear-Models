import numpy as np

def synthetic_linear_reg_data_generate(n, d, n_train, seed, noise_std=0.1):
    """
        X : Feature matrix (n, d)
        y : Noisy targets (n,)
        rw : real weights (d,)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n, d))

    # Randomly generate real weights: Scaling by 1/sqrt(d) keeps signal magnitude more stable as d changes
    rw = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d), size=d)
    
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    y = X @ rw + noise
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    return X_train, y_train, X_test, y_test, rw

def fit_least_squares(X, y): # TODO: change to self implemented function
    return np.linalg.pinv(X) @ y
def predict(X, w):
    return X @ w
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def fit_model_to_synthetic_data(model, seed_values, **data_gen_kwargs):
    mse_train = []
    mse_test = []
    for seed in seed_values:
        X_train,y_train,X_test,y_test,rw = synthetic_linear_reg_data_generate(seed=seed, **data_gen_kwargs)
        
        if model=='ls':
            w_hat = fit_least_squares(X_train, y_train)

        y_train_pred = predict(X_train, w_hat)
        y_test_pred = predict(X_test, w_hat)
        
        mse_train.append(mse(y_train, y_train_pred))
        mse_test.append(mse(y_test, y_test_pred))
    return np.mean(mse_train), np.mean(mse_test)
