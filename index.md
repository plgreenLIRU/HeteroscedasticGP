---
title: Heteroscedastic Gaussian Processes
layout: default
---

# Background

We assume that we make observations according to

$$
    y(x) = f(x) + \epsilon
$$

where $\epsilon \sim N(0, \sigma^2(x))$ i.e. the noise variance depends on $x$. We define $z$ as the log noise-variance. As well as a GP prior over the latent function

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
p(y \mid X)=\iint p(y \mid f, z) p(f \mid X)p(z \mid x) d f d z
$$

Note that our assumption is equivalent to saying that $p(z \mid f, X)=p(z \mid X)$ i.e. we're saying that the noise level depends only on the input, $X$, and not the latent function, $f$. This assumption would be invalid if, for example, we had a sensor whose noise depended on the amplitude being measured.

The likelihood is intractable because of where $z$ appears in $p(y|f,z) = N(y; f, k_f(X,X) + \text{diag}(\exp(z)))$.

Assuming that we have somehow realised maximum-likelihood estimates of the noise variance, denoted $\hat{z}$, the we would realise predictions according to

$$
\begin{aligned}
& p\left(y, \mid x_*, X, y, \hat{z}\right) \\
= & \iint p\left(y_* \mid f_*, z_*\right) p\left(f, \mid x_*, X, y\right) p\left(z_* \mid x_*, X, \hat{z}\right) d f, d z_* \\
= & \int p\left(f_* \mid x_*, X, y\right)\left[\int p\left(y_* \mid f_*, z_*\right) p\left(z_* \mid x_*, X, \hat{z}\right) d z_*\right] d f_*
\end{aligned}
$$

where it is the term in the square bracket that is intractable.

# Basic Regressor
In the BasicRegressor model we essentially ignore uncertainty in our estimate of the noise variance. The following describes how we approximate the likelihood and predictions for this case. We begin discussion looking at the case where each input is unique, before then summarising for the case where there are repeated inputs in our data.

## Log-likelihood
### Definition
To deal with the intractability in our likelihood we could fix $z=\hat{z}$ in which case we realise the approximation

$$
\begin{aligned}
p(y \mid X) & \approx p(\hat{z} \mid X) \int p(y \mid f, \hat{z})p(f \mid X) d f \\
& = N(\hat{z}; z_0, k_z(X,X)) N\left(y ; 0, k_f(X, X)+\operatorname{diag}(\exp (\hat{z}))\right)
\end{aligned}
$$

What's the best choice of $\hat{z}$ ? It seems sensible to choose the value that contributes the most to the integral with respect to $f$ in our likelihood i.e. we define $\hat{z}$ as

$$
\begin{aligned}
\hat{z}&=\underset{z}{\operatorname{argmax}} \int p(y \mid f, z) p(f \mid X) p(z \mid X) d f \\
& =\underset{z}{\operatorname{argmax}} p(z \mid X) \int p(y \mid f, z) p(f \mid X) d f \\
& =\underset{z}{\operatorname{argmax}} p(y \mid z, X) p(z \mid X) \\
& =\underset{z}{\arg \max} \left[ N\left(y ; 0, k_f(X, X)+\operatorname{diag}(\exp (z))\right) N(z ; z_0, k_z(X,X)) \right]
\end{aligned}
$$

which, setting $z=\hat{z}$, exactly recovers the original expression for our approximate likelihood. For this reason, we can define our objective function as the logarithm of the approximate likelihood and use it to optimise the parameters of our two kernels ( $\theta_f$ and $\theta_z$) as well as $\hat{z}$ i.e. we seek to maximise

$$
\begin{aligned}
\mathcal{L}\left(z, \theta_f, \theta_z\right)= & \log p\left(y \mid X, z, \theta_f\right)+\log p\left(z \mid X, \theta_z\right) \\
= & -\frac{1}{2} y^{\top}\left(k_f\left(X, X\right)+\operatorname{diag}\left(\exp(z)\right)\right)^{-1} y \\
& -\frac{1}{2} \log \left|k_f\left(X, X\right)+\operatorname{diag}\left(\exp(z)\right)\right| \\
& -\frac{1}{2} (z-z_0)^{\top} k_z\left(X, X\right)^{-1} (z-z_0)-\frac{1}{2} \log \left|k_z\left(X, X\right)\right| \\
\end{aligned}
$$

(ignoring constants). Using the notation

$$
K_f=k_f(X,X), \quad K_z = k_z(X,X)
$$

and

$$
C_y=K_f+\operatorname{diag}(\exp (z))
$$

we have

$$
\begin{aligned}
\mathcal{L}\left(z, \theta_f, \theta_z\right)= & \log p\left(y \mid X, z, \theta_f\right)+\log p\left(z \mid X, \theta_z\right) \\
= & -\frac{1}{2} y^{\top}C_y^{-1} y -\frac{1}{2} \log \left|C_y\right| \\
& -\frac{1}{2} (z-z_0)^{\top} K_z^{-1} (z-z_0)-\frac{1}{2} \log \left|K_z\right| \\
\end{aligned}
$$

### Computation
We use Cholesky decompositions to implement  Introducing

$$
\alpha_y=C_y^{-1} y, \quad \alpha_z=K_z^{-1} (z-z_0)
$$

and taking Cholesky decompositions

$$
C_y=L_y L_y^{\top}, \quad K_z=L_z L_z^{\top}
$$

we can write the log-likelihood as

$$
\begin{aligned}
\mathcal{L}\left(z, \theta_f, \theta_z\right)= & -\frac{1}{2} y^{\top} \alpha_y-\sum_{i=1}^n \log L_{y,{(i, i)}} \\
& -\frac{1}{2} (z-z_0)^{\top} \alpha_z-\sum_{i=1}^n \log L_{z,{(i, i)}} \\
\end{aligned}
$$

## Predictions
We now use a hat to denote quantities that have been calculated post-training (e.g. \( \hat{\alpha}_y \) represents \( C^{-1}_{y} y \) calculated with the kernel parameters, \( \theta_f \), set equal to their estimated maximum-likelihood value). We define \( \hat{z} \) as our estimate of the noise variance at the observed inputs. To make predictions, we must get around the intractable term in our predictive equation. We can do this by holding \( z_* \) equal to its expected value, i.e.


$$
\int p\left(y_* \mid f_*, z_*\right) p\left(z_* \mid x_*, X, \hat{z}\right) d z_*
\approx p\left(y_* \mid f_*, \hat{z}_*\right) = N(y_*; f_*, \exp(\hat{z}_*))
$$

where

$$
\hat{z}_* = \text{E}_{p\left(z_* \mid x_*, x, \hat{z}\right)}\left[z_*\right]=k_z\left(X, x_*\right)^{\top} K_z^{-1}(\hat{z} - z_0)
$$

and

$$
    \hat{\alpha}_z = K_z^{-1}(\hat{z} - z_0)
$$

Our predictive equation is then

$$
p\left(y \mid x_*, X, y, \hat{z}\right)
\approx \int p\left(y_* \mid f_*, \hat{z}_*\right) p\left(f_* \mid x_*, X, y\right) d f_*
$$

where

$$
p\left(y_* \mid f_*, \hat{z}_*\right)=N\left(y_* ; f_*, \exp \left(\hat{z}_*\right)\right)
$$

and

$$
p\left(f_* \mid x_*, X, y\right)=N(f_* ; \hat{k}_f\left(X, x_*\right)^{\top} \hat{K}_f^{-1} y,\left.\hat{k}_f\left(x_*, x_*\right)-\hat{k}_f\left(X, x_*\right)^{\top} \hat{K}_f^{-1} \hat{k}_f\left(X, x_*\right)\right)
$$

from which we find that

$$
p\left(y_* \mid x_*, x, y, \hat{z}\right) \approx N\left(y_* ; \mu_*, \sigma_*^2\right)
$$

where

$$
\mu_*=\hat{k}_f\left(X, x_*\right)^{\top} \hat{C}_y^{-1} y=\hat{k}_f\left(X, x_*\right)^{\top} \hat{\alpha}_y 
$$

$$
\sigma_x^2 = \exp \left(\hat{z}_*\right)+\hat{k}_f\left(x_*, x_*\right)
   -\hat{k}_f\left(X, x_*\right)^{\top} \hat{C}_y^{-1} \hat{k}_f\left(X, x_*\right)
$$

### Computation

Defining
$$
    \hat{\alpha}_z = K_z^{-1}(\hat{z} - z_0)
$$

we have that

$$
\hat{z}_* =k_z\left(X, x_*\right)^{\top} \hat{\alpha}_z
$$

Moreover, defining $\hat{L}_y v = \hat{k}_y(X, x_*)$ and $\hat{\alpha}_y$ as $C_y^{-1}y$ evaluated at our (estimated) maximum likelihood hyperparameters, we have

$$
\mu_*=\hat{k}_f\left(X, x_*\right)^{\top} \hat{\alpha}_y 
$$

$$
\sigma_x^2 = \exp \left(\hat{z}_*\right)+\hat{k}_f\left(x_*, x_*\right)-v^{\top} v
$$

## Non-Unique Inputs

What if are have repeated inputs i.e. $x_i=x_j$ where we expect the noise variance to be the same. Say we have $U$ unique values in $X$ and use $J_1, J_2, ... J_U$ to denote sets of indices for each unique value e.g. if


$$
X=[1,1,3,3,1]^{\top}
$$

then $U=2 \text { and } I_1=\{1,2,5\}$ and $I_2=\{3,4\}$. In general, we have that

$$
p(y \mid f, z)=\prod_{u=1}^U N\left(y_{J_u} ; f_{J_u}, I_{|J_u|} \exp \left(z_u\right)\right)
$$

$$
p(z \mid X)=N\left(z ; z_0, k_z\left(X_u, X_u\right)\right)
$$

where $X_u$ is the unique values of $X$ only. The approximate likelihood is then

$$
p(y \mid X) \approx N\left(\hat{z} ; 0, k_z\left(X_u, X_u\right)\right)
N\left(y ; O, k_f\left(X, X\right)+ \Sigma(\hat{z})\right)
$$

where $\Sigma(\hat{z})$ is diagonal with

$$
\Sigma(\hat{z})_{i,i} =
\begin{cases}
\exp(\hat{z}_1), & \text{if } i \in J_1, \\
\exp(\hat{z}_2), & \text{if } i \in J_2, \\
\;\vdots & \\[6pt]
\exp(\hat{z}_U), & \text{if } i \in J_U,
\end{cases}
$$

For predictions, we just need to change our definition of $\hat{z}_*$ to be

$$
\hat{z}_*=k_z\left(X_u, x_*\right)^{\top} k_z\left(X_u, X_u\right)^{-1} (\hat{z}-z_0)
$$
