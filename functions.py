import numpy as np

def synthetic_linear_reg_data_generate(n, d, seed, noise_std=0.1):
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

    return X, y, rw
