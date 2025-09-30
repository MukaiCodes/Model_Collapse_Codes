import numpy as np
from scipy.stats import norm
import scipy.stats as stats

def empirical_cdf_vectorized(data, x_values):
    data = np.sort(data)
    return np.searchsorted(data, x_values, side='right') / len(data)



def cramer_von_mises_statistic(data, mu, sigma, sed=1):
    np.random.seed(sed)
    U = np.random.normal(0, 1, 50000)
    F_n = empirical_cdf_vectorized(data, U)
    F_Y = norm.cdf(U, loc=mu, scale=sigma)
    return np.mean(np.square(F_n - F_Y))


def cramer_von_mises_statistic_FD(data_1, data_2, w, mu, sigma, sed=1):

    np.random.seed(sed)
    #import time
    #S = time.time()
    K = np.linspace(-5,5,50000)
    p_Y = norm.pdf(K, loc=mu, scale=sigma)
    F_Y = norm.cdf(K, loc=mu, scale=sigma)
    F_n1 = empirical_cdf_vectorized(data_1, K)
    F_n2 = empirical_cdf_vectorized(data_2, K)
    F_n = F_n1 * (1 - w) + F_n2 * w
    Out = np.dot(np.square(F_n - F_Y),p_Y)/50000*10
    #E = time.time()
    #print(E-S)

    """
    S = time.time()
    U = np.random.normal(0, 1, 300000)
    F_Y = norm.cdf(U, loc=mu, scale=sigma)
    F_n1 = empirical_cdf_vectorized(data_1, U)
    F_n2 = empirical_cdf_vectorized(data_2, U)
    F_n = F_n1 * (1 - w) + F_n2 * w
    Out = np.mean(np.square(F_n - F_Y))
    E = time.time()
    print(E-S)
    """

    return Out


"""
C = np.random.normal(0,1,[1000,1])
print(np.mean([cramer_von_mises_statistic(x, 0,1,sed=1) for x in C])*6)
for i in range(300):
    np.random.seed(i)
    n = 300
    samples = np.random.uniform(0, 1, n)
    R1 = cramer_von_mises_statistic(samples, 0, 1, sed=1)
    result = stats.cramervonmises(samples, 'norm').statistic / n
    print(R1, result)
"""
