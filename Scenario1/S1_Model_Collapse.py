import sys
sys.path.append('Scenario1')
import Frameworks.A_Fully_Synthetic as FS
import numpy as np
import pandas as pd
from seaborn import lineplot
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit
from tqdm import tqdm
# =========================
# Basic Settings
# =========================
n, p, T = 100, 4, 1000
REPS = 1000

# Result Store Location
base_dir = Path("Result_Folder")
base_dir.mkdir(parents=True, exist_ok=True)

all_temp = []

for rep in tqdm(range(REPS)):
    np.random.seed(rep)

    # ====== Real Parameter and Real Data ======
    M = np.zeros(p)
    S = (1.0 / p) * np.diag(np.ones(p))
    Beta = np.ones(p)

    # Initialize Real X
    X = np.random.multivariate_normal(M, S, n)

    # initialize Real Y
    Y_lin = (X @ Beta) + np.random.normal(0, 1, n)

    # Logistic：Generate Binary Sample
    eta_log = X @ Beta
    probs = expit(eta_log)
    Y_log = np.random.binomial(1, probs)

    # Poisson：generate poisson samples
    mu = np.exp(X @ Beta)
    Y_poi = np.random.poisson(mu)

    # CDF Estimate
    X_sample = np.random.normal(0, 1, n)

    # ====== class Application: Full Synthetic ======
    FS_class = FS.Fully_Synthetic(T)

    FS_class.Gaussian_esti(X, M, S, rep)
    FS_class.LR([X, Y_lin], Beta, rep)            # fixed design Lienar Regression
    FS_class.CDF_Esti(X_sample, rep)
    FS_class.RandomLR([X, Y_lin], Beta, rep)      # Random Design: Linear Regression
    FS_class.Logistic([X, Y_log], Beta, rep)      # Logistic（sklearn），Y_log is 0/1
    FS_class.Poisson([X, Y_poi], Beta, rep)       # Poisson（sklearn）

    # ====== Results Collections ======
    Temp = np.array([
        FS_class.loss_mean,          # Row 0
        FS_class.loss_var,           # Row 1
        FS_class.loss_beta,          # Row 2: fixed design LR
        FS_class.loss_cdf,           # Row 3
        FS_class.loss_beta_random,   # Row 4: random design LR
        FS_class.loss_logistic,      # Row 5
        FS_class.loss_poisson,       # Row 6
        [rep] * T,                   # Row 7: Rep
        list(range(T))               # Row 8: Step
    ])
    all_temp.append(Temp)



# =========================
# Merge into DataFrame and Store
# =========================
final_result = np.hstack(all_temp)  # shape: (9, REPS * T)
DF = pd.DataFrame(final_result.T, columns=[
    'E_Mean', 'E_Var', 'E_beta', 'E_CDF',
    'E_beta_random', 'E_logistic', 'E_poisson',
    'Rep', 'Step'
])

for col in ['E_Mean', 'E_Var', 'E_beta', 'E_CDF', 'E_beta_random', 'E_logistic', 'E_poisson']:
    DF[col] = pd.to_numeric(DF[col], errors='coerce')
DF['Rep'] = DF['Rep'].astype(int)
DF['Step'] = DF['Step'].astype(int)


# ==========================
# Plot Results and Store
# ==========================
def Plot_Function(output):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#ADD8E6')

    if output == '1':
        lineplot(data=DF, x='Step', y='E_Mean', ax=ax)
        plt.xlabel('Estimation Step: $t$')
        plt.ylabel('Err$(\\mathbf{\\mu}_t(0,k))$')
        plt.savefig((base_dir / "S1_Mean.png").as_posix(), dpi=300, bbox_inches='tight')
    if output == '2':
        lineplot(data=DF, x='Step', y='E_Var', ax=ax)
        plt.xlabel('Estimation Step: $t$')
        plt.ylabel('Err$(\\mathbf{\\Sigma}_t(0,k))$')
        plt.savefig((base_dir / "S1_Var.png").as_posix(), dpi=300, bbox_inches='tight')
    if output == '3':
        lineplot(data=DF, x='Step', y='E_CDF', ax=ax)
        plt.xlabel('Estimation Step: $t$')
        plt.ylabel(r'Err$(F_t(\cdot))$')  # <-- 这里改了
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        #plt.hlines(1 / 6, x_min, x_max, linestyles='--', color='red', label='$y=\\frac{1}{6}$')
        plt.ylim(y_min, y_max + 0.01)
        plt.legend(loc='upper right')
        plt.savefig((base_dir / "S1_CDF.png").as_posix(), dpi=300, bbox_inches='tight')

    if output == '4':
        lineplot(data=DF, x='Step', y='E_beta_random', ax=ax)
        plt.xlabel('Estimation Step: $t$')
        plt.ylabel('Err$(\mathbf{\\theta}_t(0,k))$ (Linear Regression)')
        plt.savefig((base_dir / "S1_BetaRandom.png").as_posix(), dpi=300, bbox_inches='tight')
    if output == '5':
        lineplot(data=DF, x='Step', y='E_logistic', ax=ax)
        plt.xlabel('Estimation Step: $t$')
        plt.ylabel('Err$(\mathbf{\\theta}_t(0,k))$ (Logistic Regression)')
        plt.savefig((base_dir / "S1_Logistic.png").as_posix(), dpi=300, bbox_inches='tight')
    if output == '6':
        lineplot(data=DF, x='Step', y='E_poisson', ax=ax)
        plt.xlabel('Estimation Step: $t$')
        plt.ylabel('Err$(\mathbf{\\theta}_t(0,k)))$ (Poisson Regression)')
        plt.savefig((base_dir / "S1_Poisson.png").as_posix(), dpi=300, bbox_inches='tight')

# Plot All Figures
for out in ['1', '2', '3', '4', '5', '6']:
    Plot_Function(out)
