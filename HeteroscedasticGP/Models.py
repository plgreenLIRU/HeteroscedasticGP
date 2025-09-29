import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

class BasicRegressor:

    def __init__(self, ARD):

        self.ARD = ARD

    def find_gram_matrix(self, X: np.ndarray, params: dict, X_star: np.ndarray = None):
        """
        Compute Gram (kernel) matrix between X and X_star under RBF kernel.

        Args:
            X: (N, D) array of training inputs
            params: dict with "lengthscale" (or "lengthscale i" if ARD)
            X_star: (M, D) array of test inputs (optional). 
                    If None, computes K(X, X).

        Returns:
            K: (N, M) kernel matrix if X_star provided, else (N, N).
        """
        if X_star is None:
            X_star = X

        N, D = np.shape(X)
        M, D_star = np.shape(X_star)

        assert D == D_star, "X and X_star must have the same number of features"

        if not self.ARD:
            # Single shared lengthscale
            ls = params["lengthscale"]
            squared_dist = cdist(X, X_star, metric="sqeuclidean")
            K = np.exp(-0.5 / (ls**2) * squared_dist)

        else:
            # Initialise log-kernel matrix
            K = np.zeros([N, M])

            # Loop over input dimensions
            for i in range(D):
                # Extract lengthscale for dimension i
                ls = params[f"lengthscale {i+1}"]

                # Compute squared distances for ith feature between X and X_star
                squared_dist = cdist(
                    np.atleast_2d(X[:, i]).T, 
                    np.atleast_2d(X_star[:, i]).T, 
                    metric="sqeuclidean"
                )

                # Update kernel accumulator
                K -= 0.5 / (ls**2) * squared_dist

            # Exponentiate to get Gram matrix
            K = np.exp(K)

        return K


    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray, z: np.ndarray,
                           f_params: dict, z_params: dict):
        Cy = self.find_gram_matrix(X, params=f_params) + np.diag(np.exp(z)) + 1e-6 * np.eye(len(y))
        Kz = self.find_gram_matrix(X, params=z_params) + 1e-6 * np.eye(len(y))

        Ly = np.linalg.cholesky(Cy)
        Lz = np.linalg.cholesky(Kz)

        alpha_y = cho_solve((Ly, True), y)
        alpha_z = cho_solve((Lz, True), z)

        neg_logl = (0.5 * y.T @ alpha_y + np.log(np.diag(Ly)).sum() +
                    0.5 * z.T @ alpha_z + np.log(np.diag(Lz)).sum())

        return neg_logl

    def _pack_params(self, f_params: dict, z_params: dict, z: np.ndarray) -> np.ndarray:
        """
        Function to take parameter dictionaries & latent vector z and create
        a corresponding numpy array.
        """
        theta = np.concatenate([np.array(list(f_params.values()), dtype=float), np.array(list(z_params.values()), dtype=float), z.ravel()])    
        return theta

    def _unpack_params(self, theta: np.ndarray, f_keys, z_keys, z_dim: int):
        """
        Function to convert parameter vector theta back into dictionaries
        """
        f_size = len(f_keys)
        z_size = len(z_keys)        
        f_params = {k: v for k, v in zip(f_keys, theta[:f_size])}
        z_params = {k: v for k, v in zip(z_keys, theta[f_size:f_size+z_size])}
        z = theta[f_size+z_size:].reshape(z_dim,)
        return f_params, z_params, z

    def _objective(self, theta, X, y, f_keys, z_keys, z_dim):
        """
        Called during training; takes array as inputs then converts it
        to a dictionary for the neg_log_likelihood function
        """
        f_params, z_params, z = self._unpack_params(theta, f_keys, z_keys, z_dim)
        return self.neg_log_likelihood(X, y, z, f_params, z_params)

    def train(self, X, y, f_params0, z_params0, z0):
        """
        Train model based on initial guess of f parameters, z parameters, and
        initial guess of the array, z.
        """

        # Find unique rows in X
        Xu, inverse_indices, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)

        # Determine whether or not we are looking at a problem with repeated inputs
        if np.all(counts == 1):
            self.repeated_X = False
        else:
            self.repeated_X = True
            self.Xu = Xu

        # If repeated X, create J_list that stores indices of unique X values
        self.J_list = []
        if self.repeated_X == True:
            for u in range(len(Xu)):
                idx = np.where(inverse_indices == u)[0]
                self.J_list.append(idx)            
        
        theta0 = self._pack_params(f_params0, z_params0, z0)
        res = minimize(self._objective, theta0, args=(X, y, list(f_params0.keys()), list(z_params0.keys()), z0.shape[0]), method="L-BFGS-B")
        f_params, z_params, z_opt = self._unpack_params(res.x, f_params0.keys(), z_params0.keys(), z0.shape[0])
        self.assign_hyperparameters(X, y, f_params, z_params, z_opt)

    def assign_hyperparameters(self, X, y, f_params, z_params, z_opt):

        # Assign training data and optimal hyperparameters
        self.X = X
        self.y = y
        self.f_params_opt = f_params
        self.z_params_opt = z_params
        self.z_opt = z_opt

        # Compute matrices and vectors needed for predictions
        self.Cy = self.find_gram_matrix(X, params=f_params) + np.diag(np.exp(z_opt)) + 1e-6 * np.eye(len(y))
        Kz = self.find_gram_matrix(X, params=z_params) + 1e-6 * np.eye(len(y))
        self.Ly = np.linalg.cholesky(self.Cy)
        Lz = np.linalg.cholesky(Kz)
        self.alpha_y = cho_solve((self.Ly, True), y)
        self.alpha_z = cho_solve((Lz, True), z_opt)

    def predict(self, X_star):
        
        # Evaluate kernels evaluated between training and sprediction inputs
        K_f_star = self.find_gram_matrix(X=self.X, params=self.f_params_opt, X_star=X_star)
        K_z_star = self.find_gram_matrix(X=self.X, params=self.z_params_opt, X_star=X_star)

        # Preditive z mean
        z_star = K_z_star.T @ self.alpha_z
        
        # Predictive y mean
        mu_star = K_f_star.T @ self.alpha_y

        # Predictive variance
        v = solve_triangular(self.Ly, K_f_star, lower=True)
        var_star = np.exp(z_star) + 1 - np.diag(v.T @ v)

        return mu_star, var_star
