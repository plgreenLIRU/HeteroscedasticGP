import numpy as np

class BasicRegressor:

    def __init__(self):

        pass

    def find_gram_matrix(self, theta):
        """
        Description
        -----------
            Method that computes the gram matrix (not
                including noise terms)

        Parameters
        ----------
            theta : numpy array of GP hyperparameters

        Returns
        -------
            K : gram matrix
        """

        if self.ARD is False:
                ls = theta[0]
                squared_dist = cdist(self.X, self.X, metric='sqeuclidean')
                K = np.exp(-1 / (2 * ls**2) * squared_dist)

        else:

            # Initialise N by N array of zeros. Ultimately, in the
            # following code, we will evaluate K = exp(S) to
            # get the gram matrix.
            K = np.zeros([self.N, self.N])

            # Loop over inputs
            for i in range(self.D):

                # Length scale for the ith input
                ls = theta[i]

                # Calculate squared distances for ith input
                # (which has to be a 2D array).
                squared_dist = cdist(np.vstack(self.X[:, i]),
                                        np.vstack(self.X[:, i]),
                                        metric='sqeuclidean')

                # Multiply squared distance by -1/(2 ls**2)
                K -= 1 / (2 * ls**2) * squared_dist

            # Compute gram matrix
            K = np.exp(K)

        return K

    def neg_log_likelihood(self, theta):

        # Compute gram matrix
        sigma = theta[-1]
        C = self.find_gram_matrix(theta) + np.eye(self.N) * sigma**2

        # Find Cholesky decomposition (such that C=LL^T)
        L = np.linalg.cholesky(C)

        # Cholesky solve for alpha (C alpha = y)
        alpha = cho_solve((L, True), self.Y)

        # Evaluate negative log-likelihood; note that log|C| is the
        # summation of diagonal terms in L
        nll = (0.5 * self.Y.T @ alpha + np.log(np.diag(L)).sum())[0][0]

        return nll

    def train(self):

        # Function that the solver will call to print theta per solver iteration
        def solver_callback(x):            
            print('lengthscales = ', np.round(x[:-1] * 100) / 100)
            print('noise std = ', x[-1])

        # Run solver
        sol = minimize(self.neg_log_likelihood, x0=self.initial_theta, method='SLSQP', bounds=self.theta_bounds, callback=solver_callback)

        # Assign parameter estimates and solution outcome
        theta = sol.x
        self.sol = sol
        self.nlogp = sol.fun
        self.assign_hyperparameters(theta)

    def assign_hyperparameters(self, theta):

        sigma = theta[-1]
        C = self.find_gram_matrix(theta) + np.eye(self.N) * sigma**2
        self.theta = theta
        self.L = np.linalg.cholesky(C)
        self.alpha = cho_solve((self.L, True), self.Y)

    def predict(self, X_star):
        pass
