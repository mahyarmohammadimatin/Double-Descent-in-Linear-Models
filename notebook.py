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
from functions import *
import numpy as np

# %%
# General Parameters Setup
n_train = 100
n_test = 1000
noise_values = [0.5]
dim_values = np.arange(10, 201, 5) # Data Dimension
seed_values = np.arange(1,3) # fixed seeds to see the same result as me

# %% [markdown]
# ## Simulation on Least Squares

# %%
model = 'ls'
simulation = DDSimulation(model=model, n_train=n_train, n_test=n_test,
			 dim_values=dim_values, seed_values=seed_values, noise_values=noise_values)
simulation.run_simulation()

# %% [markdown]
# ## Simulation on Ridge Regression

# %%
model = 'ridge'
lam_values = [1e-2]
simulation = DDSimulation(model=model, n_train=n_train, n_test=n_test,
			 dim_values=dim_values, seed_values=seed_values, noise_values=noise_values, lam_values=lam_values)
simulation.run_simulation()
