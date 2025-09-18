import numpy as np

class BasicRegressor:

    def __init__(self):

        pass

def find_gram_matrix(self, params: dict):
    """
    Compute Gram matrix for Gaussian Process.

    Parameters
    ----------
    params : dict
        Dictionary containing kernel hyperparameters.
        - If ARD is False: expects {"lengthscale": value}.
        - If ARD is True: expects {"lengthscale 1": v1, "lengthscale 2": v2, ..., "lengthscale D": vD}.
    """

    if not self.ARD:
        # Single shared lengthscale
        ls = params["lengthscale"]
        squared_dist = cdist(self.X, self.X, metric="sqeuclidean")
        K = np.exp(-0.5 / (ls**2) * squared_dist)

    else:
        # Initialise log-kernel matrix
        K = np.zeros([self.N, self.N])

        # Loop over input dimensions
        for i in range(self.D):
            # Extract lengthscale for dimension i
            ls = params[f"lengthscale {i+1}"]

            # Compute squared distances for ith feature
            squared_dist = cdist(
                np.atleast_2d(self.X[:, i]).T,
                np.atleast_2d(self.X[:, i]).T,
                metric="sqeuclidean"
            )

            # Update kernel accumulator
            K -= 0.5 / (ls**2) * squared_dist

        # Exponentiate to get Gram matrix
        K = np.exp(K)

    return K


    def neg_log_likelihood(self, params: dict):

        pass

    def train(self, X, Y):

        # Run solver
        sol = minimize(self.neg_log_likelihood, x0=self.initial_theta, method='SLSQP', bounds=self.theta_bounds)

        # Assign parameter estimates and solution outcome
        theta = sol.x
        self.sol = sol
        self.nlogp = sol.fun
        self.assign_hyperparameters(theta)

    def assign_hyperparameters(self, params: dict):

        pass

    def predict(self, X_star):
        pass
