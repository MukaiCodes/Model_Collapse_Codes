import sys
sys.path.append('Scenario3')
import numpy as np
from multiprocessing import Pool
import pandas as pd
import Frameworks.D_Fresh_Data as DFD
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import lineplot
from scipy.special import expit

# ========================
# Global Configuration
# ========================
N, P, START, T = 100, 2, 150,200
NUM_PROCS      = 10
CHUNK_SIZE     = 120
REPS           = 1000
K_LIST = np.round(np.arange(0.01, 0.2, 0.02), 4)

OUT_DIR = Path("Result_Folder")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "results_S3.csv"

HEADER = [
    "Final_E_Mean", "Final_E_Var",
    "Final_E_beta_random",
    "Final_E_logistic", "Final_E_poisson",
    "Final_E_CDF",
    "k", "rep", "scheme", "w_used"
]
if not OUT_CSV.exists():
    pd.DataFrame(columns=HEADER).to_csv(OUT_CSV, index=False)

# =============== Weighting Formulas ===============
def w_naive_from_k(k: float) -> float:
    return k / (1.0 + k)   # naive weighting

def w_opt_closed_form(k: float) -> float:
    return (np.sqrt(k*k + 4.0*k) - k) / 2.0   # optimal weighting

# =============== Evaluate All Models for a Given w ===============
def eval_all_models_for_w(w: float, k_eff: float, rep: int, data_pack):
    (X_set, RD_input, LOG_input, POI_input, D_cdf_set,
     M, S, Beta) = data_pack

    fd = DFD.Fresh_Data(T, w, k_eff)

    # Multivariate Gaussian estimation
    fd.Gaussian_esti(X_set, M, S, seed=rep, start=START)

    # Random-design Linear Regression
    fd.RandomLR(RD_input, Beta, seed=rep, start=START, noise_std=1.0)

    # Logistic Regression
    fd.Logistic(LOG_input, Beta, seed=rep, start=START, max_iter=200)

    # Poisson Regression
    fd.Poisson(POI_input, Beta, seed=rep, start=START, max_iter=300)

    # CDF Estimation
    fd.CDF_Esti(D_cdf_set, seed=rep, start=START)

    return dict(
        mean   = getattr(fd, "Final_loss_mean",   np.nan),
        var    = getattr(fd, "Final_loss_var",    np.nan),
        beta_r = getattr(fd, "Final_loss_beta_random", np.nan),
        logit  = getattr(fd, "Final_loss_logistic",    np.nan),
        pois   = getattr(fd, "Final_loss_poisson",     np.nan),
        cdf    = getattr(fd, "Final_loss_cdf",         np.nan),
    )

# =============== Single (k, rep) Task ===============
def S3_case(k_in: float, rep: int):
    if k_in <= 0:
        raise ValueError("k must be > 0")
    k_eff = k_in

    np.random.seed(rep)
    M = np.zeros(P)
    S = np.eye(P)
    Beta = np.ones(P)

    # X_t ~ N(M,S)
    X_set = [np.random.multivariate_normal(M, S, N) for _ in range(T)]

    # Random-design Linear Regression data
    Y_lin_rd_seq = [(X @ Beta) + np.random.normal(0, 1, N) for X in X_set]
    RD_input     = [(X_set[t], Y_lin_rd_seq[t]) for t in range(T)]

    # Logistic Regression data
    eta_log_seq = [X @ Beta for X in X_set]
    probs_seq   = [expit(eta) for eta in eta_log_seq]
    Y_log_seq   = [np.random.binomial(1, probs) for probs in probs_seq]
    LOG_input   = [(X_set[t], Y_log_seq[t]) for t in range(T)]

    # Poisson Regression data
    eta_poi_seq = [np.clip(X @ Beta, -20.0, 20.0) for X in X_set]
    mu_seq      = [np.exp(eta) for eta in eta_poi_seq]
    Y_poi_seq   = [np.random.poisson(mu) for mu in mu_seq]
    POI_input   = [(X_set[t], Y_poi_seq[t]) for t in range(T)]

    # CDF data
    D_cdf_set = [np.random.normal(0.0, 1.0, N) for _ in range(T)]

    data_pack = (X_set, RD_input, LOG_input, POI_input, D_cdf_set, M, S, Beta)

    # Naive weighting
    w_nv = w_naive_from_k(k_in)
    res_nv = eval_all_models_for_w(w_nv, k_eff, rep, data_pack)
    row_nv = [
        res_nv["mean"], res_nv["var"], res_nv["beta_r"],
        res_nv["logit"], res_nv["pois"], res_nv["cdf"],
        k_in, rep, "naive", w_nv
    ]

    # Optimal weighting
    w_op = w_opt_closed_form(k_in)
    res_op = eval_all_models_for_w(w_op, k_eff, rep, data_pack)
    row_op = [
        res_op["mean"], res_op["var"], res_op["beta_r"],
        res_op["logit"], res_op["pois"], res_op["cdf"],
        k_in, rep, "optimal", w_op
    ]

    return [row_nv, row_op]

# =============== Plotting ===============
def plot_s3_results_each(csv_file, out_dir):
    df = pd.read_csv(csv_file)
    for c in ["Final_E_Mean","Final_E_Var","Final_E_beta_random",
              "Final_E_logistic","Final_E_poisson","Final_E_CDF","k","w_used"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df['scheme'] == 'naive', 'scheme'] = 'naive mixing'
    df.loc[df['scheme'] == 'optimal', 'scheme'] = 'optimal weighting'

    df_mean = df.groupby(["k","scheme"], as_index=False).mean(numeric_only=True)

    models = {
        "Final_E_Mean":        ("Gaussian Mean", "S3_Mean.png"),
        "Final_E_Var":         ("Gaussian Variance", "S3_Var.png"),
        "Final_E_beta_random": ("Linear Regression (Random)", "S3_LR.png"),
        "Final_E_logistic":    ("Logistic Regression", "S3_Logistic.png"),
        "Final_E_poisson":     ("Poisson Regression", "S3_Poisson.png"),
        "Final_E_CDF":         ("CDF Distance", "S3_CDF.png"),
    }

    for col,(ylabel,filename) in models.items():
        fig, ax = plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#ADD8E6")
        lineplot(data=df_mean, x="k", y=col, hue="scheme", style="scheme", ax=ax,markers='o')
        ax.set_xlabel(r"Ratio $k = n/m$")
        ax.set_ylabel('Estimation Error')
        ax.legend(title="Methods", loc="best")
        out_path = Path(out_dir)/filename
        plt.savefig(out_path,dpi=300,bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {out_path.as_posix()}")

# =============== Main Program ===============
if __name__=="__main__":
    t0 = time.time()
    Cases=[(float(k),int(rep)) for k in K_LIST for rep in range(REPS)]
    with Pool(processes=NUM_PROCS) as pool:
        for i in tqdm(range(0,len(Cases),CHUNK_SIZE)):
            batch=Cases[i:i+CHUNK_SIZE]
            batch_results=pool.starmap(S3_case,batch)
            rows=[row for two in batch_results for row in two]
            pd.DataFrame(rows,columns=HEADER).to_csv(OUT_CSV,mode="a",header=False,index=False)
            if (i//CHUNK_SIZE)%10==0:
                print(f"Progress: {i+len(batch):,}/{len(Cases):,}")
    print(f"Elapsed time: {time.time()-t0:.2f} seconds")
    print(f"Saved to: {OUT_CSV.as_posix()}")
    plot_s3_results_each(OUT_CSV,OUT_DIR)
