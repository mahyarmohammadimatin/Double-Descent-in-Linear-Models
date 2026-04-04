from utils import *
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

class SyntheticData:
    def __init__(self, n, dim, seed=1):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.dim = dim
        self.data = self.rng.normal(loc=0.0, scale=1.0, size=(n, dim))
    def generate_linear_regression_data(self, noise_std=0.1):
        """
            X (data) : Feature matrix (n, d)
            y : Noisy targets (n,)
            rw : real weights (d,)
        """
        X = self.data
        # Randomly generate real weights: Scaling by 1/sqrt(d) keeps signal magnitude more stable as d changes
        rw = self.rng.normal(loc=0.0, scale=1.0 / np.sqrt(self.dim), size=self.dim)
        noise = self.rng.normal(loc=0.0, scale=noise_std, size=self.n)
        y = X @ rw + noise
        return X, y
    @staticmethod
    def split_train_test(X, y, n_train):
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
        return X_train, y_train, X_test, y_test

class DDSimulation:
    def __init__(self, model, n_train, n_test, dim_values, seed_values, noise_values, lam_values):
        # Fixed Attrs
        self.model = model
        self.n_train = n_train
        self.n_test = n_test

        # Variable Attrs
        self.dim_values = dim_values
        self.seed_values = seed_values
        self.noise_values = noise_values
        self.lam_values = lam_values

    def fit_model_to_specific_config(self, seed, dim, noise_std, lam=None):
        synthetic_data = SyntheticData(n=self.n_train+self.n_test, dim=dim, seed=seed)
        X, y = synthetic_data.generate_linear_regression_data(noise_std=noise_std)
        X_train, y_train, X_test, y_test = synthetic_data.split_train_test(X, y, n_train=self.n_train)

        if self.model == 'ls':
            w_hat = fit_least_squares(X_train, y_train)
        elif self.model == 'ridge':
            w_hat = fit_ridge_regression(X_train, y_train, lam)
        else:
            raise ValueError("model must be 'ls' or 'ridge'")

        y_train_pred = predict(X_train, w_hat)
        y_test_pred = predict(X_test, w_hat)

        return mse(y_train, y_train_pred), mse(y_test, y_test_pred)

    def run_simulation(self):
        results = {}
        for noise_std, lam in product(self.noise_values, self.lam_values):
            train_errors, test_errors = [], []
            for dim in self.dim_values:
                mse_trains, mse_tests = [], []
                for seed in self.seed_values:
                    mse_train, mse_test = self.fit_model_to_specific_config(seed=seed, dim=dim, noise_std=noise_std, lam=lam)
                    mse_trains.append(mse_train)
                    mse_tests.append(mse_test)
                # Average on different seeds
                train_errors.append(np.mean(mse_trains))
                test_errors.append(np.mean(mse_tests))
            # for each combination of (noise_std, lam) save list of errors corresponding to dimension values
            results[f'train_{noise_std}_{lam}'] = train_errors
            results[f'test_{noise_std}_{lam}'] = test_errors

        # self.plot_simulation(results) TODO: add in future for all combinations of (noise_std, lam)
        self.plot_train_test_error(train_errors, test_errors)

    def plot_train_test_error(self, train_errors, test_errors):
        plt.figure(figsize=(8, 5))
        plt.plot(self.dim_values, train_errors, label="Train error")
        plt.plot(self.dim_values, test_errors, label="Test error")
        plt.axvline(x=self.n_train, linestyle="--", label=f"Interpolation threshold d=n={self.n_train}")
        plt.xlabel("Model complexity (dimension d)")
        plt.ylabel("Mean squared error")
        plt.title("Least Squares: Train/Test Error vs Model Complexity")
        plt.legend()
        plt.tight_layout()
        plt.show()