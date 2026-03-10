# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# !jupytext --sync notebook.ipynb

# %%
from functions import fit_model_to_synthetic_data
import numpy as np

# %%
# General Parameters Setup
n_train = 100
n_test = 1000
noise_std = 0.5
d_values = np.arange(10, 201, 5)
seed_values = np.arange(1,3) # fixed seeds to see the same result as me

# %%
train_errors = []
test_errors = []

for d in d_values:
    mse_train,mse_test = fit_model_to_synthetic_data(model='ls', seed_values=seed_values,
                                n=n_train+n_test, d=d, n_train=n_train, noise_std=noise_std)
    train_errors.append(mse_train)
    test_errors.append(mse_test)


# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(d_values, train_errors, label="Train error")
plt.plot(d_values, test_errors, label="Test error")
plt.axvline(x=n_train, linestyle="--", label=f"Interpolation threshold d=n={n_train}")
plt.xlabel("Model complexity (dimension d)")
plt.ylabel("Mean squared error")
plt.title("Least Squares: Train/Test Error vs Model Complexity")
plt.legend()
plt.tight_layout()
plt.show()

# %%
