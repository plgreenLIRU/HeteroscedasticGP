---
title: Heteroscedastic Gaussian Processes
layout: default
---

# Background

We assume that we make observations according to

$$
y(x) = f(x) + \epsilon
$$

where $\epsilon \sim N(0, \sigma^2(x))$, i.e. the noise variance depends on $x$. We define $z$ as the log noise-variance. As well as a GP prior over the latent function

$$
p(f|X) = N(f; 0, k_f(X,X))
$$

we also define a GP prior over the log noise-variance

$$
p(z|X) = N(z; z_0, k_z(X, X))
$$

where $k_f(\cdot, \cdot)$ has hyperparameters $\theta_f$ and $k_z(\cdot, \cdot)$ has hyperparameters $\theta_z$ and $z_0$ is the mean of our Gaussian Process over $z$.

## Maximum likelihood

To get a maximum likelihood estimate of the model's hyperparameters we would, ideally, maximise

$$
p(y \mid X)=\iint p(y \mid f, z) p(f, z \mid X) d f d z
$$

If we assume that $p(f, z \mid X)=p(f \mid X) p(z \mid X)$ then we can write the likelihood as

$$
p(y \mid X)=\iint p(y \mid f, z) p(f \mid X)p(z \mid X) d f d z
$$

Note that our assumption is equivalent to saying that $p(z \mid f, X)=p(z \mid X)$, i.e., we're saying that the noise level depends only on the input, $X$, and not the latent function, $f$.

The likelihood is intractable because of where $z$ appears in $p(y|f,z)=N(y;f,k_f(X,X)+\text{diag}(\exp(z)))$.

Assuming that we have somehow realised maximum-likelihood estimates of the noise variance, denoted $\hat{z}$, we would realise predictions according to

$$
\begin{aligned}
p(y_* \mid x_*, X, y, \hat{z}) 
&= \iint p(y_* \mid f_*, z_*)\, p(f_* \mid x_*, X, y)\, p(z_* \mid x_*, X, \hat{z}) \, df_* dz_* \\
&= \int p(f_* \mid x_*, X, y) \Big[ \int p(y_* \mid f_*, z_*)\, p(z_* \mid x_*, X, \hat{z}) dz_* \Big] df_*
\end{aligned}
$$

where it is the term in square brackets that is intractable.

# Basic Regressor

In the BasicRegressor model we essentially ignore uncertainty in our estimate of the noise variance. The following describes how we approximate the likelihood and predictions for this case. We begin by discussing the case where each input is unique, before summarising the case with repeated inputs.

## Log-likelihood

### Definition

To deal with the intractability in our likelihood, we could fix $z=\hat{z}$, in which case:

$$
\begin{aligned}
p(y \mid X) &\approx p(\hat{z} \mid X) \int p(y \mid f, \hat{z})p(f \mid X) df \\
&= N(\hat{z}; z_0, k_z(X,X)) \, N\left(y ; 0, k_f(X, X)+\operatorname{diag}(\exp (\hat{z}))\right)
\end{aligned}
$$

We define $\hat{z}$ as:

$$
\begin{aligned}
\hat{z} &= \underset{z}{\operatorname{argmax}} \int p(y \mid f, z) p(f \mid X) p(z \mid X) df \\
&= \underset{z}{\operatorname{argmax}} p(z \mid X) \int p(y \mid f, z) p(f \mid X) df \\
&= \underset{z}{\operatorname{argmax}} p(y \mid z, X) p(z \mid X) \\
&= \underset{z}{\arg \max} \Big[ N\left(y ; 0, k_f(X, X)+\operatorname{diag}(\exp (z))\right) N(z ; z_0, k_z(X,X)) \Big]
\end{aligned}
$$

We define our objective function as the logarithm of the above and maximise with respect to $\theta_f$, $\theta_z$, and $\hat{z}$:

$$
\begin{aligned}
\mathcal{L}(z, \theta_f, \theta_z) &= \log p(y \mid X, z, \theta_f) + \log p(z \mid X, \theta_z) \\
&= -\frac{1}{2} y^{\top} (k_f(X, X) + \operatorname{diag}(\exp(z)))^{-1} y \\
&\quad - \frac{1}{2} \log |k_f(X, X) + \operatorname{diag}(\exp(z))| \\
&\quad - \frac{1}{2} (z-z_0)^{\top} k_z(X, X)^{-1} (z-z_0) - \frac{1}{2} \log |k_z(X, X)|
\end{aligned}
$$

We define:

$$
K_f = k_f(X,X), \quad K_z = k_z(X,X), \quad C_y = K_f + \operatorname{diag}(\exp(z))
$$

Then the log-likelihood can also be written as:

$$
\mathcal{L}(z, \theta_f, \theta_z) = -\frac{1}{2} y^{\top} C_y^{-1} y - \frac{1}{2} \log |C_y| - \frac{1}{2} (z-z_0)^{\top} K_z^{-1} (z-z_0) - \frac{1}{2} \log |K_z|
$$

### Computation

We introduce:

$$
\alpha_y = C_y^{-1} y, \quad \alpha_z = K_z^{-1} (z-z_0)
$$

and Cholesky decompositions:

$$
C_y = L_y L_y^{\top}, \quad K_z = L_z L_z^{\top}
$$

Then the log-likelihood is:

$$
\mathcal{L}(z, \theta_f, \theta_z) = -\frac{1}{2} y^{\top} \alpha_y - \sum_{i=1}^n \log L_{y,{(i, i)}} - \frac{1}{2} (z-z_0)^{\top} \alpha_z - \sum_{i=1}^n \log L_{z,{(i, i)}}
$$

## Predictions

Define $\hat{z}$ as our estimate of the noise variance at observed inputs. We approximate:

$$
\int p(y_* \mid f_*, z_*) p(z_* \mid x_*, X, \hat{z}) dz_* \approx p(y_* \mid f_*, \hat{z}_*) = N(y_*; f_*, \exp(\hat{z}_*))
$$

with

$$
\hat{z}_* = \text{E}_{p(z_* \mid x_*, X, \hat{z})}[z_*] = k_z(X, x_*)^\top K_z^{-1} (\hat{z} - z_0)
$$

and

$$
\hat{\alpha}_z = K_z^{-1} (\hat{z} - z_0)
$$

The predictive equation:

$$
p(y_* \mid x_*, X, y, \hat{z}) \approx \int p(y_* \mid f_*, \hat{z}_*) p(f_* \mid x_*, X, y) df_*
$$

where

$$
p(y_* \mid f_*, \hat{z}_*) = N(y_*; f_*, \exp(\hat{z}_*))
$$

$$
p(f_* \mid x_*, X, y) = N(f_*; \hat{k}_f(X, x_*)^\top \hat{K}_f^{-1} y, \hat{k}_f(x_*, x_*) - \hat{k}_f(X, x_*)^\top \hat{K}_f^{-1} \hat{k}_f(X, x_*))
$$

Finally:

$$
p(y_* \mid x_*, X, y, \hat{z}) \approx N(y_*; \mu_*, \sigma_*^2)
$$

$$
\mu_* = \hat{k}_f(X, x_*)^\top \hat{\alpha}_y, \quad
\sigma_x^2 = \exp(\hat{z}_*) + \hat{k}_f(x_*, x_*) - \hat{k}_f(X, x_*)^\top \hat{C}_y^{-1} \hat{k}_f(X, x_*)
$$

### Non-Unique Inputs

For repeated inputs $x_i = x_j$, define $U$ unique values with index sets $J_1, ..., J_U$. E.g.

$$
X = [1, 1, 3, 3, 1]^\top
$$

then $U=2$ with $I_1=\{1,2,5\}$, $I_2=\{3,4\}$. Then

$$
p(y \mid f, z) = \prod_{u=1}^U N(y_{J_u}; f_{J_u}, I_{|J_u|} \exp(z_u))
$$

$$
p(z \mid X) = N(z; z_0, k_z(X_u, X_u))
$$

with approximate likelihood:

$$
p(y \mid X) \approx N(\hat{z}; 0, k_z(X_u, X_u)) N(y; 0, k_f(X, X) + \Sigma(\hat{z}))
$$

where

$$
\Sigma(\hat{z})_{i,i} =
\begin{cases}
\exp(\hat{z}_1), & i \in J_1 \\
\exp(\hat{z}_2), & i \in J_2 \\
\vdots \\
\exp(\hat{z}_U), & i \in J_U
\end{cases}
$$

Predictions use:

$$
\hat{z}_* = k_z(X_u, x_*)^\top k_z(X_u, X_u)^{-1} (\hat{z} - z_0)
$$

