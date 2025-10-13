import numpy as np
from HeteroscedasticGP.Models import BaseGP


def test_1D():
    """
    Test the BaseGP model on 1D sinusoidal data with homoscedastic noise.
    Checks that predictions are close to the true function and noise variance.
    """
    np.random.seed(42)

    # Generate training and test data
    N = 200
    N_star = 500
    X = np.linspace(0, 10, N)
    X_star = np.linspace(0, 10, N_star)
    f = np.sin(X)
    f_star = np.sin(X_star)
    noise_var_true = 0.05
    y = f + np.sqrt(noise_var_true) * np.random.randn(N)

    # Initialize GP model
    gp = BaseGP(ARD=False)

    # Set initial kernel parameters and noise variance
    f_params0 = {'scale': 1.0, 'lengthscale': 1.0}
    noise_var0 = 0.1

    # Train GP model
    gp.train(np.vstack(X), y, f_params0=f_params0, noise_var0=noise_var0)

    # Predict on test points
    mu_star_gp, var_star_gp = gp.predict(np.vstack(X_star))

    # Check that mean predictions are close to true function
    assert np.allclose(mu_star_gp, f_star, atol=0.15)

    # Estimate noise variance from residuals and check closeness
    estimated_noise_var = np.var(y - gp.predict(np.vstack(X))[0])
    assert np.allclose(estimated_noise_var, noise_var_true, atol=0.01)

    # Check that predicted variance is close to true noise variance
    assert np.allclose(var_star_gp, noise_var_true, atol=0.01)
