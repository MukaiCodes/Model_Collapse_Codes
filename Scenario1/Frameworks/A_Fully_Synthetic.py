import numpy as np
import Frameworks.Base_function as BF
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from scipy.special import expit

class Fully_Synthetic:
    def __init__(self, T):
        self.T = T
        self.loss_mean = [0 for i in range(self.T)]
        self.loss_var = [0 for i in range(self.T)]
        self.loss_beta = [0 for i in range(self.T)]   # for fixed-design linear regression
        self.loss_cdf = [0 for i in range(self.T)]
        self.loss_beta_random = [0 for i in range(self.T)]
        self.loss_logistic = [0 for i in range(self.T)]
        self.loss_poisson = [0 for i in range(self.T)]

    # =========================
    # Existing: Gaussian mean/variance (keep as is)
    # =========================
    def Gaussian_esti(self, D_0, mean, cov, seed=1):
        size, p = D_0.shape
        mean_t = np.mean(D_0, axis=0)
        var_t = np.cov(D_0, rowvar=False).reshape(p, p)
        self.loss_mean[0] = np.sum((mean_t - mean) ** 2)
        self.loss_var[0] = np.sum((var_t - cov) ** 2)
        np.random.seed(seed)
        for t in range(1, self.T):
            synthe = np.random.multivariate_normal(mean_t, var_t, size)
            mean_t = np.mean(synthe, axis=0)
            var_t = np.cov(synthe, rowvar=False).reshape(p, p)
            self.loss_mean[t] = np.linalg.norm(mean_t - mean) ** 2
            self.loss_var[t] = np.linalg.norm(var_t - cov) ** 2

    # =========================
    # Existing: Fixed-design Linear Regression (keep as is)
    # =========================
    def LR(self, D_0, beta, seed=1):
        # n, p = 100, 4
        X, Y = D_0
        n, p = X.shape
        Comp = (np.linalg.inv(X.T @ X) @ X.T)
        Beta_hat = (Comp * Y).sum(axis=1)
        self.loss_beta[0] = np.linalg.norm(Beta_hat - beta) ** 2
        np.random.seed(seed)
        for t in range(1, self.T):
            Beta_hat = (Comp * Y).sum(axis=1)
            Y = (X * Beta_hat).sum(axis=1) + np.random.normal(0, 1, n)
            self.loss_beta[t] = np.linalg.norm(Beta_hat - beta) ** 2

    # =========================
    # Existing: CDF Estimation (keep as is)
    # =========================
    def CDF_Esti(self, X, seed=1):
        n = len(X)
        self.loss_cdf[0] = BF.cramer_von_mises_statistic(X, 0, 1, sed=1)
        for t in range(1, self.T):
            np.random.seed(seed * t)
            X = np.random.choice(X, n)
            Error_F = BF.cramer_von_mises_statistic(X, 0, 1, sed=1)
            self.loss_cdf[t] = Error_F

    # =========================================================
    # New Method 1: Random-design Linear Regression (OLS, Scenario 1)
    # =========================================================
    def RandomLR(self, D_0, beta, seed=1, noise_std=1.0):
        """
        Fully Synthetic + Random Design Linear Regression
        Each iteration: X ~ N(0, I),  Y = X @ beta_prev + eps, eps ~ N(0, noise_std^2)
        Estimation: OLS (np.linalg.lstsq)
        Record: self.loss_beta_random[t] = ||beta_hat - beta||^2
        Do not modify or reset existing loss arrays; only update loss_beta_random.
        """
        X0, Y0 = D_0
        X0 = np.asarray(X0, dtype=float)
        Y0 = np.asarray(Y0, dtype=float).ravel()
        n, p = X0.shape
        if beta.shape[0] != p:
            raise ValueError(f"Dimension of beta {beta.shape[0]} does not match number of columns in X {p}.")

        np.random.seed(seed)

        # t = 0: estimate from initial real data
        beta_hat = np.linalg.lstsq(X0, Y0, rcond=None)[0]
        self.loss_beta_random[0] = float(np.linalg.norm(beta_hat - beta) ** 2)

        # t >= 1: fully synthetic recursive training
        for t in range(1, self.T):
            X = np.random.normal(loc=0.0, scale=1.0, size=(n, p))
            eps = np.random.normal(loc=0.0, scale=noise_std, size=n)
            Y = X @ beta_hat + eps

            beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
            self.loss_beta_random[t] = float(np.linalg.norm(beta_hat - beta) ** 2)

    # =========================================================
    # New Method 2: Logistic Regression (sklearn, Scenario 1)
    # =========================================================
    def Logistic(self, D_0, beta, seed=1, max_iter=2000):
        """
        Fully Synthetic + Random Design Logistic Regression
        Each iteration: X ~ N(0, I),  Y ~ Bernoulli(sigmoid(X @ beta_prev))
        Estimation: sklearn LogisticRegression (no intercept, prefer no regularization,
                    fallback to very weak L2 regularization if needed)
        Record: self.loss_logistic[t] = ||beta_hat - beta||^2
        To include intercept, add a column of ones to X and extend beta accordingly.
        """
        X0, Y0 = D_0
        X0 = np.asarray(X0, dtype=float)
        Y0 = np.asarray(Y0).ravel()
        n, p = X0.shape
        if beta.shape[0] != p:
            raise ValueError(f"Dimension of beta {beta.shape[0]} does not match number of columns in X {p}.")

        # Ensure Y0 is in {0,1}
        uniq = np.unique(Y0)
        if not np.array_equal(uniq, [0]) and not np.array_equal(uniq, [1]) and not np.array_equal(uniq, [0, 1]):
            # If Y0 is probability or score, threshold it into {0,1}
            Y0 = (Y0 > 0.5).astype(int)

        np.random.seed(seed)

        # t = 0: estimate from initial real data
        beta_hat = self._fit_logistic(X0, Y0, max_iter=max_iter)
        self.loss_logistic[0] = float(np.linalg.norm(beta_hat - beta) ** 2)

        # t >= 1: fully synthetic recursive training
        for t in range(1, self.T):
            X = np.random.normal(0.0, 1.0, size=(n, p))
            eta = X @ beta_hat
            probs = expit(eta)
            Y = np.random.binomial(1, probs)

            beta_hat = self._fit_logistic(X, Y, max_iter=max_iter)
            self.loss_logistic[t] = float(np.linalg.norm(beta_hat - beta) ** 2)

    def _fit_logistic(self, X, y, max_iter=200):
        """
        Internal utility: robust logistic fitting;
        Prefer MLE with penalty='none'; fallback to very weak L2 if unsupported.
        """
        try:
            clf = LogisticRegression(
                penalty='none', solver='lbfgs', fit_intercept=False, max_iter=max_iter
            )
            clf.fit(X, y)
        except Exception:
            clf = LogisticRegression(
                penalty='l2', C=1e6, solver='lbfgs', fit_intercept=False, max_iter=max_iter
            )
            clf.fit(X, y)
        return clf.coef_.ravel()

    # =========================================================
    # New Method 3: Poisson Regression (sklearn, Scenario 1)
    # =========================================================
    def Poisson(self, D_0, beta, seed=1, max_iter=300):
        """
        Fully Synthetic + Random Design Poisson Regression
        Each iteration: X ~ N(0, I),  Y ~ Poisson(exp(X @ beta_prev))
        Estimation: sklearn PoissonRegressor (alpha=0 disables regularization; no intercept)
        Record: self.loss_poisson[t] = ||beta_hat - beta||^2
        """
        X0, Y0 = D_0
        X0 = np.asarray(X0, dtype=float)
        Y0 = np.asarray(Y0, dtype=float).ravel()
        if np.any(Y0 < 0):
            raise ValueError("Poisson regression requires non-negative response variable y.")
        n, p = X0.shape
        if beta.shape[0] != p:
            raise ValueError(f"Dimension of beta {beta.shape[0]} does not match number of columns in X {p}.")

        np.random.seed(seed)

        # t = 0: estimate from initial real data
        pr = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=max_iter)
        pr.fit(X0, Y0)
        beta_hat = pr.coef_.ravel()
        self.loss_poisson[0] = float(np.linalg.norm(beta_hat - beta) ** 2)

        # t >= 1: fully synthetic recursive training
        for t in range(1, self.T):
            X = np.random.normal(0.0, 1.0, size=(n, p))
            eta = X @ beta_hat
            eta = np.clip(eta, -20.0, 20.0)  # prevent overflow in exp
            mu = np.exp(eta)
            Y = np.random.poisson(mu)

            pr = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=max_iter)
            pr.fit(X, Y)
            beta_hat = pr.coef_.ravel()
            self.loss_poisson[t] = float(np.linalg.norm(beta_hat - beta) ** 2)
