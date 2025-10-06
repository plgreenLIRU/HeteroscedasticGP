import numpy as np
from HeteroscedasticGP.Models import BasicRegressor

def test_repeated_x():
    ''' Test example where we have repated inputs '''

    # Fixing seed for example
    np.random.seed(42)

    # Inputs
    n_repeat = 100
    X = np.repeat(0, n_repeat)
    X = np.append(X, np.repeat(1, n_repeat))
    X = np.append(X, np.repeat(2, n_repeat))
    X = np.vstack(X)
    true_var = np.repeat(0.3, n_repeat)
    true_var = np.append(true_var, np.repeat(1, n_repeat))
    true_var = np.append(true_var, np.repeat(1.5, n_repeat))

    # True function
    f_true = np.sin(X).ravel()

    # Generate noisy outputs
    y = f_true + true_var * np.random.randn(len(X))

    # True over new inputs
    X_star = np.vstack(np.linspace(0, 2, 50))
    f_star_true = np.sin(X_star).ravel()

    # Extract true z
    z_true = np.log(np.unique(true_var))

    # Fit model
    m = BasicRegressor(ARD=False)
    f_params0 = {'lengthscale': 1}
    z_params0 = {'lengthscale': 1}
    z0 = np.zeros(3)
    m.train(X, y, f_params0=f_params0, z_params0=z_params0, z0=z0)

    # Check we have detected repetaed inputs
    assert m.repeated_X == True
    assert m.U == 3

    # Check indices of repeated points
    assert np.array_equal(m.J_list[0], np.arange(0, n_repeat))
    assert np.array_equal(m.J_list[1], np.arange(n_repeat, n_repeat*2))
    assert np.array_equal(m.J_list[2], np.arange(n_repeat*2, n_repeat*3))

    # Check noise std
    assert np.allclose(np.sqrt(np.exp(z_true)), np.sqrt(np.exp(m.z_opt)), atol=0.5)

    # Make predictions
    mu_star, var_star = m.predict(X_star)

    # Check predictions (mean)
    np.allclose(mu_star, f_star_true, atol=0.2)