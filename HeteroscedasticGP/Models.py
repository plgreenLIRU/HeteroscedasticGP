import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

class BaseGP:

    def __init__(self, ARD):
        """
        Initialise Gaussian Process

        Args:
            ARD (bool): If True, use Automatic Relevance Determination (ARD) for kernel lengthscales.
        """
        self.ARD = ARD

    def _find_gram_matrix(self, X: np.ndarray, params: dict, X_star: np.ndarray=None):
        """
        Compute Gram (kernel) matrix between X and X_star under RBF kernel.

        Args:
            X: (N, D) array of training inputs
            params: dict with "scale" and "lengthscale" (or "lengthscale i" if ARD)
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
            K = params["scale"] * np.exp(-0.5 / (ls**2) * squared_dist)

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
            K = params["scale"] * np.exp(K)

        return K
    
    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray, f_params: dict, noise_var):
        """
        Compute the negative log-likelihood for the heteroscedastic GP model.

        Args:
            X: Training inputs.
            y: Training targets.
            z: Latent log-variance values.
            f_params: Kernel parameters for function.
            noise_var : noise_variance

        Returns:
            neg_logl: Negative log-likelihood value.
        """
        Cy = self._find_gram_matrix(X, params=f_params) + np.eye(len(y)) * (noise_var + 1e-6)
        Ly = np.linalg.cholesky(Cy)
        alpha_y = cho_solve((Ly, True), y)
        neg_logl = 0.5 * y.T @ alpha_y + np.log(np.diag(Ly)).sum()
        return neg_logl

    def _pack_params(self, f_params: dict, noise_var: float):
        """
        Combine f_params and noise variance into a single 1D parameter array.

        Handles both scalar and vector-valued parameters (e.g., ARD lengthscales).
        Returns flattened parameter array plus metadata needed for unpacking.
        """
        f_keys = list(f_params.keys())
        f_shapes = [np.shape(np.atleast_1d(f_params[k])) for k in f_keys]

        # Flatten and concatenate
        f_values = np.concatenate([np.atleast_1d(f_params[k]).ravel() for k in f_keys])

        # Append noise variance as a single scalar
        theta = np.concatenate([f_values, np.atleast_1d(noise_var)])

        return theta, f_keys, f_shapes

    def _unpack_params(self, theta: np.ndarray, f_keys, f_shapes):
        """
        Convert flattened parameter vector theta back into f_params and noise variance.

        Args:
            theta: Flattened parameter array.
            f_keys: Keys for function parameters.
            f_shapes: Shapes for each parameter.

        Returns:
            f_params: Dictionary of unpacked parameters.
            noise_var: Scalar noise variance.
        """
        f_params = {}
        idx = 0

        # Rebuild f_params
        for k, shape in zip(f_keys, f_shapes):
            size = np.prod(shape)
            value = theta[idx:idx + size].reshape(shape)
            f_params[k] = value.item() if value.size == 1 else value
            idx += size

        # Remaining element is noise variance
        noise_var = float(theta[idx])

        return f_params, noise_var
    
    def _objective(self, theta, X, y, f_keys, f_shapes):
        """
        Objective function for optimizer. Converts parameter array to dictionaries and computes negative log-likelihood.

        Args:
            theta: Flattened parameter array.
            X, y: Training data.
            f_keys, f_shapes: Metadata for unpacking.
        Returns:
            Negative log-likelihood value.
        """
        f_params, noise_var = self._unpack_params(theta, f_keys, f_shapes)
        return self.neg_log_likelihood(X, y, f_params, noise_var)

    def train(self, X, y, f_params0, noise_var0):
        """
        Train the heteroscedastic GP model by optimizing hyperparameters and latent variables.

        Args:
            X: Training inputs.
            y: Training targets.
            f_params0: Initial function kernel parameters.
            noise_var0 : Initial noise variance.
        """

        # Pack initial parameters into single array for optimisation
        theta0, f_keys, f_shapes = self._pack_params(f_params0, noise_var0)

        # Optimise
        res = minimize(self._objective, theta0, args=(X, y, f_keys, f_shapes), method="L-BFGS-B")

        # Assign optimal hyperparameters
        f_params, noise_var = self._unpack_params(res.x, f_keys, f_shapes)
        self.assign_hyperparameters(X, y, f_params, noise_var)

    def assign_hyperparameters(self, X, y, f_params, noise_var):
        """
        Assign optimized hyperparameters and latent variables to the model.

        Args:
            X: Training inputs.
            y: Training targets.
            f_params: Optimized function kernel parameters.
            noise_var : noise variance for likelihood
        """

        # Assign training data and hyperparameters
        self.X = X
        self.y = y
        self.f_params_opt = f_params
        self.noise_var = noise_var

        self.Cy = self._find_gram_matrix(X, params=f_params) + np.eye(len(y)) * (noise_var + 1e-6)
        self.Ly = np.linalg.cholesky(self.Cy)
        self.alpha_y = cho_solve((self.Ly, True), y)

    def predict(self, X_star):
        """
        Predict mean and variance for new inputs.

        Args:
            X_star: Test inputs.

        Returns:
            mu_star: Predictive mean.
            var_star: Predictive variance.
        """
        
        # Evaluate kernels evaluated between training and sprediction inputs
        K_f_star = self._find_gram_matrix(X=self.X, params=self.f_params_opt, X_star=X_star)

        # Predictive y mean
        mu_star = K_f_star.T @ self.alpha_y

        # Predictive variance
        v = solve_triangular(self.Ly, K_f_star, lower=True)
        var_star = self.noise_var + 1 - np.diag(v.T @ v)

        return mu_star, var_star

class BasicRegressor(BaseGP):

    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray, z: np.ndarray,
                           f_params: dict, z_params: dict, z0_mean: float):
        """
        Compute the negative log-likelihood for the heteroscedastic GP model.

        Args:
            X: Training inputs.
            y: Training targets.
            z: Latent log-variance values.
            f_params: Kernel parameters for function.
            z_params: Kernel parameters for variance.
            z0_mean: Mean of latent log-variance.

        Returns:
            neg_logl: Negative log-likelihood value.
        """
        # If we don't have any repeated X values
        if self.repeated_X == False:
            Cy = self._find_gram_matrix(X, params=f_params) + np.diag(np.exp(z)) + 1e-6 * np.eye(len(y))
            Kz = self._find_gram_matrix(X, params=z_params) + 1e-6 * np.eye(len(z))

        # If we do have repeated X values
        if self.repeated_X == True:

            # Form covariance matrix
            sigma2 = np.zeros([len(y)])
            for u in range(self.U):
                Ju = self.J_list[u]
                sigma2[Ju] = np.exp(z[u])
            Sigma2 = np.diag(sigma2)

            Cy = self._find_gram_matrix(X, params=f_params) + Sigma2 + 1e-6 * np.eye(len(y))
            Kz = self._find_gram_matrix(self.Xu, params=z_params) + 1e-6 * np.eye(len(z))

        Ly = np.linalg.cholesky(Cy)
        Lz = np.linalg.cholesky(Kz)

        alpha_y = cho_solve((Ly, True), y)
        alpha_z = cho_solve((Lz, True), z - z0_mean)

        neg_logl = (0.5 * y.T @ alpha_y + np.log(np.diag(Ly)).sum() +
                    0.5 * (z - z0_mean).T @ alpha_z + np.log(np.diag(Lz)).sum())

        return neg_logl

    def _pack_params(self, f_params: dict, z_params: dict, z: np.ndarray):
        """
        Combine f_params, z_params, and latent vector z into a single 1D parameter array.
        Handles both scalar and vector-valued parameters (e.g., ARD lengthscales).
        Returns flattened parameter array plus metadata needed for unpacking.
        """
        f_keys = list(f_params.keys())
        z_keys = list(z_params.keys())

        # Record shapes (so ARD params can be reconstructed)
        f_shapes = [np.shape(np.atleast_1d(f_params[k])) for k in f_keys]
        z_shapes = [np.shape(np.atleast_1d(z_params[k])) for k in z_keys]

        # Flatten and concatenate
        f_values = np.concatenate([np.atleast_1d(f_params[k]).ravel() for k in f_keys])
        z_values = np.concatenate([np.atleast_1d(z_params[k]).ravel() for k in z_keys])
        theta = np.concatenate([f_values, z_values, z.ravel()])

        return theta, f_keys, z_keys, f_shapes, z_shapes

    def _unpack_params(self, theta: np.ndarray, f_keys, z_keys, f_shapes, z_shapes, z_dim: int):
        """
        Convert flattened parameter vector theta back into dictionaries and latent vector z.
        Shapes for each parameter are provided so ARD parameters are restored correctly.

        Args:
            theta: Flattened parameter array.
            f_keys, z_keys: Keys for function and variance parameters.
            f_shapes, z_shapes: Shapes for each parameter.
            z_dim: Dimension of latent vector z.

        Returns:
            f_params, z_params, z: Unpacked parameters and latent vector.
        """
        f_params = {}
        z_params = {}
        idx = 0

        # Rebuild f_params
        for k, shape in zip(f_keys, f_shapes):
            size = np.prod(shape)
            value = theta[idx:idx + size].reshape(shape)
            f_params[k] = value.item() if value.size == 1 else value
            idx += size

        # Rebuild z_params
        for k, shape in zip(z_keys, z_shapes):
            size = np.prod(shape)
            value = theta[idx:idx + size].reshape(shape)
            z_params[k] = value.item() if value.size == 1 else value
            idx += size

        # Remaining are latent z values
        z = theta[idx:].reshape(z_dim,)

        return f_params, z_params, z

    def _objective(self, theta, X, y, f_keys, z_keys, f_shapes, z_shapes, z_dim, z0_mean):
        """
        Objective function for optimizer. Converts parameter array to dictionaries and computes negative log-likelihood.

        Args:
            theta: Flattened parameter array.
            X, y: Training data.
            f_keys, z_keys, f_shapes, z_shapes: Metadata for unpacking.
            z_dim: Dimension of latent vector z.
            z0_mean: Mean of latent log-variance.

        Returns:
            Negative log-likelihood value.
        """
        f_params, z_params, z = self._unpack_params(theta, f_keys, z_keys, f_shapes, z_shapes, z_dim)
        return self.neg_log_likelihood(X, y, z, f_params, z_params, z0_mean)

    def train(self, X, y, f_params0, z_params0, z0, z0_mean):
        """
        Train the heteroscedastic GP model by optimizing hyperparameters and latent variables.

        Args:
            X: Training inputs.
            y: Training targets.
            f_params0: Initial function kernel parameters.
            z_params0: Initial variance kernel parameters.
            z0: Initial latent log-variance.
            z0_mean: Mean of latent log-variance.
        """

        # Identify if we have repeated X values
        self._identify_repeated_X(X)

        # Pack initial parameters into single array for optimisation
        theta0, f_keys, z_keys, f_shapes, z_shapes = self._pack_params(f_params0, z_params0, z0)

        # Optimise
        res = minimize(self._objective, theta0, args=(X, y, f_keys, z_keys, f_shapes, z_shapes, z0.shape[0], z0_mean), method="L-BFGS-B")

        # Assign optimal hyperparameters
        f_params, z_params, z_opt = self._unpack_params(res.x, f_keys, z_keys, f_shapes, z_shapes, z0.shape[0])
        self.assign_hyperparameters(X, y, f_params, z_params, z_opt, z0_mean)

    def _identify_repeated_X(self, X):
        """
        Identify repeated input values in X and set up indexing for unique values.

        Args:
            X: Input data.
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
        if self.repeated_X == True:
            self.J_list = []
            for u in range(len(Xu)):
                idx = np.where(inverse_indices == u)[0]
                self.J_list.append(idx)
            self.U = len(self.J_list)

    def assign_hyperparameters(self, X, y, f_params, z_params, z_opt, z0_mean):
        """
        Assign optimized hyperparameters and latent variables to the model.

        Args:
            X: Training inputs.
            y: Training targets.
            f_params: Optimized function kernel parameters.
            z_params: Optimized variance kernel parameters.
            z_opt: Optimized latent log-variance.
            z0_mean: Mean of latent log-variance.
        """

        # Assign training data and hyperparameters
        self.X = X
        self.y = y
        self.f_params_opt = f_params
        self.z_params_opt = z_params
        self.z_opt = z_opt
        self.z0_mean = z0_mean

        # Identify if we have repeated X values
        self._identify_repeated_X(X)        

        # If we don't have any repeated X values
        if self.repeated_X == False:
            self.Cy = self._find_gram_matrix(X, params=f_params) + np.diag(np.exp(z_opt)) + 1e-6 * np.eye(len(y))
            Kz = self._find_gram_matrix(X, params=z_params) + 1e-6 * np.eye(len(z_opt))

        # If we do have repeated X values
        if self.repeated_X == True:

            # Form covariance matrix
            sigma2 = np.zeros([len(y)])
            for u in range(self.U):
                Ju = self.J_list[u]
                sigma2[Ju] = np.exp(z_opt[u])
            Sigma2 = np.diag(sigma2)

            self.Cy = self._find_gram_matrix(X, params=f_params) + Sigma2 + 1e-6 * np.eye(len(y))
            Kz = self._find_gram_matrix(self.Xu, params=z_params) + 1e-6 * np.eye(len(z_opt))

        self.Ly = np.linalg.cholesky(self.Cy)
        Lz = np.linalg.cholesky(Kz)
        self.alpha_y = cho_solve((self.Ly, True), y)
        self.alpha_z = cho_solve((Lz, True), z_opt - self.z0_mean)

    def predict(self, X_star):
        """
        Predict mean and variance for new inputs.

        Args:
            X_star: Test inputs.

        Returns:
            mu_star: Predictive mean.
            var_star: Predictive variance.
            z_star: Predicted latent log-variance.
        """
        
        # Evaluate kernels evaluated between training and sprediction inputs
        K_f_star = self._find_gram_matrix(X=self.X, params=self.f_params_opt, X_star=X_star)
        if self.repeated_X == False:
            K_z_star = self._find_gram_matrix(X=self.X, params=self.z_params_opt, X_star=X_star)
        if self.repeated_X == True:
            K_z_star = self._find_gram_matrix(X=self.Xu, params=self.z_params_opt, X_star=X_star)

        # Preditive z mean
        z_star = self.z0_mean + K_z_star.T @ self.alpha_z
        
        # Predictive y mean
        mu_star = K_f_star.T @ self.alpha_y

        # Predictive variance
        v = solve_triangular(self.Ly, K_f_star, lower=True)
        var_star = np.exp(z_star) + 1 - np.diag(v.T @ v)

        return mu_star, var_star, z_star
