import numpy as np
import Frameworks.Base_function as BF
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from scipy.special import expit
class Fresh_Data:
    def __init__(self,T,w,k):
        self.T = T
        self.w = w
        self.k = k
        self.loss_mean = [0 for _ in range(self.T)]
        self.loss_var  = [0 for _ in range(self.T)]
        self.loss_beta = [0 for _ in range(self.T)]
        self.loss_cdf  = [0 for _ in range(self.T)]
        # 新增三类的误差记录
        self.loss_beta_random = [0 for _ in range(self.T)]
        self.loss_logistic    = [0 for _ in range(self.T)]
        self.loss_poisson     = [0 for _ in range(self.T)]

    # =========================
    # 一维高斯（保持原样）
    # =========================
    def Uni_Gaussian_esti(self,D_0,mean,cov,seed=1,start=500):
        size = len(D_0[0])
        mean_t = np.mean(D_0[0])
        var_t  = np.var(D_0[0], ddof=1)
        self.loss_mean[0] = (mean_t - mean) ** 2
        self.loss_var[0]  = (var_t  - cov ) ** 2
        np.random.seed(seed)
        Syn_size = max(1, int(size/self.k))
        for t in range(1, self.T):
            np.random.seed(self.T * seed + t)
            synthe = np.random.normal(mean_t, np.sqrt(var_t), Syn_size)
            mean_real = np.mean(D_0[t])
            Var_real  = np.var(D_0[t], ddof=1)
            mean_t = (1-self.w) * np.mean(synthe) + self.w * mean_real
            var_t  = (1-self.w) * np.var(synthe, ddof=1) + self.w * Var_real
            if t >= start:
                self.loss_mean[t] = (mean_t - mean) ** 2
                self.loss_var[t]  = (var_t  - cov ) ** 2
        self.Final_loss_mean = np.mean(self.loss_mean[start::])
        self.Final_loss_var  = np.mean(self.loss_var[start::])

    # =========================
    # 多元高斯（保持原样）
    # =========================
    def Gaussian_esti(self,D_0,mean,cov,seed=1,start=500):
        size, p = D_0[0].shape
        mean_t = np.mean(D_0[0], axis=0)
        var_t  = np.cov(D_0[0], rowvar=False).reshape(p, p)
        self.loss_mean[0] = np.sum((mean_t - mean) ** 2)
        self.loss_var[0]  = np.sum((var_t  - cov ) ** 2)
        np.random.seed(seed)
        Syn_size = max(1, int(size/self.k))
        for t in range(1, self.T):
            np.random.seed(self.T * seed + t)
            synthe = np.random.multivariate_normal(mean_t, var_t, Syn_size)
            mean_real = np.mean(D_0[t], axis=0)
            Var_real  = np.cov(D_0[t], rowvar=False).reshape(p, p)
            mean_t = (1-self.w) * np.mean(synthe, axis=0) + self.w * mean_real
            var_t  = (1-self.w) * np.cov(synthe, rowvar=False).reshape(p, p) + self.w * Var_real
            if t >= start:
                self.loss_mean[t] = np.linalg.norm(mean_t - mean)**2
                self.loss_var[t]  = np.linalg.norm(var_t  - cov )**2
        self.Final_loss_mean = np.mean(self.loss_mean[start::])
        self.Final_loss_var  = np.mean(self.loss_var[start::])

    # =========================
    # fixed design 线回归
    # =========================
    def LR(self, D_0, beta, seed=1, start=500):
        """
        Fixed-design Linear Regression in Fresh Data Framework.
        - X 固定不变（来自 D_0[0][0]）
        - 每轮只在响应 Y 上做 fresh/synthetic convex combination
        - w 控制 real/synthetic 比例
        """
        X, Y = D_0[0]  # 初始的 X 和 Y
        n = len(Y)
        Comp = (np.linalg.inv(X.T @ X) @ X.T)  # (X^T X)^{-1} X^T
        Beta_temp = (Comp * Y).sum(axis=1)  # 初始估计
        self.loss_beta[0] = np.linalg.norm(Beta_temp - beta) ** 2

        np.random.seed(seed)
        for t in range(1, self.T):
            # 合成数据：用上一步 Beta_temp 回归生成
            Y = (X * Beta_temp).sum(axis=1) + np.random.normal(0, 1, n)

            # 混合：synthetic Y 和 real Y 的 convex combination
            Beta_temp = (Comp * ((1 - self.w) * Y + self.w * D_0[t][1])).sum(axis=1)

            # 只在 start 之后记录误差
            if t >= start:
                self.loss_beta[t] = np.linalg.norm(Beta_temp - beta) ** 2

        # 取平均作为 Final loss
        self.Final_loss_beta = np.mean(self.loss_beta[start::])

    # =========================
    # CDF 估计（保持原样）
    # =========================
    def CDF_Esti(self, D, seed=1, start=500):
        """
        Fresh Data 的非参数 CDF 估计（考虑 k = n/m）：
          - 真实池每轮大小为 n = len(D[t])
          - 合成池大小 m = max(1, floor(n / self.k))
          - 抽样时对每个样本使用 per-sample 权重：w/n 与 (1-w)/m
        """
        # 初始真实样本规模
        n_real = len(D[0])
        # 合成池规模：m = n / k（至少为 1）
        m_syn = max(1, int(n_real / self.k))

        # 初始化“合成池”X_syn：用第0轮真实样本自举生成 m_syn 个
        # 也可以按你偏好改成从某目标分布产生；这里与现有逻辑保持接近
        rng = np.random.default_rng(seed)
        X_syn = rng.choice(D[0], size=m_syn, replace=True)

        # 误差数组第0项按需要记录（可选）
        # 如果你希望与原版保持一致，也可以先不记
        # self.loss_cdf[0] = BF.cramer_von_mises_statistic_FD(X_syn, D[0], self.w, 0, 1, sed=1)

        for t in range(1, self.T):
            np.random.seed(seed * t)

            # 1) 在评估阶段，比较 (1-w)*F_syn + w*F_real 与 N(0,1) 的 C-v-M 距离
            if t >= start:
                err = BF.cramer_von_mises_statistic_FD(X_syn, D[t], self.w, 0, 1, sed=1)
                self.loss_cdf[t] = err

            # 2) 形成下一轮的“合成池”（规模仍为 m_syn）
            #    拼接真实池与当前合成池，并按 per-sample 权重抽样 m_syn 个
            real = np.asarray(D[t])
            syn = np.asarray(X_syn)
            combo = np.concatenate([real, syn])

            # 每样本概率：真实样本权重 = w/n；合成样本权重 = (1-w)/m
            w_real_per = self.w / max(1, len(real))
            w_syn_per = (1.0 - self.w) / max(1, len(syn))

            prob = np.concatenate([
                np.full(len(real), w_real_per),
                np.full(len(syn), w_syn_per)
            ])
            prob = prob / prob.sum()

            X_syn = rng.choice(combo, size=m_syn, replace=True, p=prob)

        # 取 start..T-1 的平均作为最终误差
        self.Final_loss_cdf = np.mean(self.loss_cdf[start:])

    # =========================================================
    # 新增：Random-design Linear Regression (per-sample weights)
    # =========================================================
    def RandomLR(self, D_seq, beta, seed=1, start=500, noise_std=1.0):
        """
        D_seq: 长度 T 的 [(X_t, Y_t)]。每轮：
          1) 合成 (X_syn,Y_syn)：X_syn ~ N(0,I), Y_syn = X_syn @ beta_prev + eps
          2) 与 fresh (X_t,Y_t) 合并，按每样本权重：(1-w)/m 与 w/n，做加权 OLS
        """
        X0, Y0 = D_seq[0]
        X0 = np.asarray(X0, dtype=float); Y0 = np.asarray(Y0, dtype=float).ravel()
        n0, p = X0.shape
        syn_n0 = max(1, int(n0/self.k))

        # t=0: 仅用 fresh 估计
        beta_hat = np.linalg.lstsq(X0, Y0, rcond=None)[0]
        self.loss_beta_random[0] = float(np.linalg.norm(beta_hat - beta)**2)

        np.random.seed(seed)
        for t in range(1, self.T):
            np.random.seed(self.T * seed + t)

            # --- 合成数据 ---
            X_syn = np.random.normal(0.0, 1.0, size=(syn_n0, p))
            eps   = np.random.normal(0.0, noise_std, size=syn_n0)
            Y_syn = X_syn @ beta_hat + eps

            # --- fresh 数据 ---
            Xr, Yr = D_seq[t]
            Xr = np.asarray(Xr, dtype=float); Yr = np.asarray(Yr, dtype=float).ravel()
            n_real, m_syn = Xr.shape[0], X_syn.shape[0]

            # per-sample 权重
            w_real_per = self.w / max(1, n_real)
            w_syn_per  = (1 - self.w) / max(1, m_syn)

            # 加权 OLS：对行做 sqrt(weight) 缩放
            Xc = np.vstack([
                np.sqrt(w_syn_per)  * X_syn,
                np.sqrt(w_real_per) * Xr
            ])
            yc = np.concatenate([
                np.sqrt(w_syn_per)  * Y_syn,
                np.sqrt(w_real_per) * Yr
            ])

            beta_hat = np.linalg.lstsq(Xc, yc, rcond=None)[0]
            if t >= start:
                self.loss_beta_random[t] = float(np.linalg.norm(beta_hat - beta)**2)

        self.Final_loss_beta_random = np.mean(self.loss_beta_random[start::])

    # =========================================================
    # 新增：Logistic Regression (per-sample weights)
    # =========================================================
    def Logistic(self, D_seq, beta, seed=1, start=500, max_iter=200):
        """
        每轮：
          1) 合成 (X_syn, Y_syn)：X_syn ~ N(0,I), Y_syn ~ Bernoulli(sigmoid(X_syn @ beta_prev))
          2) 与 fresh 合并，sample_weight = [(1-w)/m]*m + [w/n]*n
        """
        X0, Y0 = D_seq[0]
        X0 = np.asarray(X0, dtype=float); Y0 = np.asarray(Y0).ravel()
        n0, p = X0.shape
        syn_n0 = max(1, int(n0/self.k))

        # 兜底：确保 {0,1}
        uniq = np.unique(Y0)
        if not np.array_equal(uniq, [0]) and not np.array_equal(uniq, [1]) and not np.array_equal(uniq, [0, 1]):
            Y0 = (Y0 > 0.5).astype(int)

        # t=0：仅用 fresh
        beta_hat = self._fit_logistic_fd(X0, Y0, sample_weight=None, max_iter=max_iter)
        self.loss_logistic[0] = float(np.linalg.norm(beta_hat - beta)**2)

        np.random.seed(seed)
        for t in range(1, self.T):
            np.random.seed(self.T * seed + t)

            # 合成
            X_syn = np.random.normal(0.0, 1.0, size=(syn_n0, p))
            eta=X_syn @ beta_hat
            probs = expit(eta)
            Y_syn = np.random.binomial(1, probs)

            # fresh
            Xr, Yr = D_seq[t]
            Xr = np.asarray(Xr, dtype=float); Yr = np.asarray(Yr).ravel()
            uniq = np.unique(Yr)
            if not np.array_equal(uniq, [0]) and not np.array_equal(uniq, [1]) and not np.array_equal(uniq, [0, 1]):
                Yr = (Yr > 0.5).astype(int)

            m_syn, n_real = X_syn.shape[0], Xr.shape[0]
            w_real_per = self.w / max(1, n_real)
            w_syn_per  = (1 - self.w) / max(1, m_syn)

            Xc = np.vstack([X_syn, Xr])
            yc = np.concatenate([Y_syn, Yr])
            sw = np.concatenate([
                np.full(m_syn, w_syn_per, dtype=float),
                np.full(n_real, w_real_per, dtype=float)
            ])

            beta_hat = self._fit_logistic_fd(Xc, yc, sample_weight=sw, max_iter=max_iter)
            if t >= start:
                self.loss_logistic[t] = float(np.linalg.norm(beta_hat - beta)**2)

        self.Final_loss_logistic = np.mean(self.loss_logistic[start::])

    def _fit_logistic_fd(self, X, y, sample_weight=None, max_iter=2000):
        """稳健 Logistic 拟合：penalty='none' 优先，失败回退极弱 L2；支持 sample_weight。"""
        try:
            clf = LogisticRegression(
                penalty=None, solver='lbfgs', fit_intercept=False, max_iter=max_iter
            )
            clf.fit(X, y, sample_weight=sample_weight)
        except Exception:
            clf = LogisticRegression(
                penalty='l2', C=1e6, solver='lbfgs', fit_intercept=False, max_iter=max_iter
            )
            clf.fit(X, y, sample_weight=sample_weight)
        return clf.coef_.ravel()

    # =========================================================
    # 新增：Poisson Regression (per-sample weights)
    # =========================================================
    def Poisson(self, D_seq, beta, seed=1, start=500, max_iter=2000):
        """
        每轮：
          1) 合成 (X_syn,Y_syn)：X_syn ~ N(0,I), Y_syn ~ Poisson(exp(X_syn @ beta_prev))
          2) 与 fresh 合并，sample_weight = [(1-w)/m]*m + [w/n]*n
        """
        X0, Y0 = D_seq[0]
        X0 = np.asarray(X0, dtype=float); Y0 = np.asarray(Y0, dtype=float).ravel()
        if np.any(Y0 < 0):
            raise ValueError("Poisson 回归要求 y 非负。")
        n0, p = X0.shape
        syn_n0 = max(1, int(n0/self.k))

        # t=0：仅用 fresh
        pr = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=max_iter)
        pr.fit(X0, Y0)
        beta_hat = pr.coef_.ravel()
        self.loss_poisson[0] = float(np.linalg.norm(beta_hat - beta)**2)

        np.random.seed(seed)
        for t in range(1, self.T):
            np.random.seed(self.T * seed + t)

            # 合成
            X_syn = np.random.normal(0.0, 1.0, size=(syn_n0, p))
            eta   = np.clip(X_syn @ beta_hat, -20.0, 20.0)
            mu    = np.exp(eta)
            Y_syn = np.random.poisson(mu)

            # fresh
            Xr, Yr = D_seq[t]
            Xr = np.asarray(Xr, dtype=float); Yr = np.asarray(Yr, dtype=float).ravel()
            if np.any(Yr < 0):
                raise ValueError("Poisson 回归要求 y 非负（fresh 数据含负值）。")

            m_syn, n_real = X_syn.shape[0], Xr.shape[0]
            w_real_per = self.w / max(1, n_real)
            w_syn_per  = (1 - self.w) / max(1, m_syn)

            Xc = np.vstack([X_syn, Xr])
            yc = np.concatenate([Y_syn, Yr])
            sw = np.concatenate([
                np.full(m_syn, w_syn_per, dtype=float),
                np.full(n_real, w_real_per, dtype=float)
            ])

            pr = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=max_iter)
            pr.fit(Xc, yc, sample_weight=sw)
            beta_hat = pr.coef_.ravel()
            if t >= start:
                self.loss_poisson[t] = float(np.linalg.norm(beta_hat - beta)**2)

        self.Final_loss_poisson = np.mean(self.loss_poisson[start::])

    # =========================
    # 汇总接口（保持原样）
    # =========================
    #def Last_Round_Error(self):
        #Output = [self.Final_loss_mean, self.Final_loss_var, self.Final_loss_beta]
        #return Output
