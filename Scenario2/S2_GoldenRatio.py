import sys
sys.path.append('Scenario2')
import numpy as np
from multiprocessing import Pool
import pandas as pd
import Frameworks.D_Fresh_Data as DFD
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import expit
# ========================
# Single (w, rep) task
# ========================
def S2_w(w, rep):
    n, p, Start, T = 100, 2, 150, 200

    # Fix the random seed (reproducible for each rep)
    np.random.seed(rep)

    # True distribution parameters
    M = np.zeros(p)
    S = np.eye(p)
    Beta = np.ones(p)

    # ====== Fresh data sequence (for random design) ======
    # X_t ~ N(M, S)
    X_set = [np.random.multivariate_normal(M, S, n) for _ in range(T)]

    # ----- Random-design linear regression: use own (X_t, Y_t) for each round -----
    Y_lin_rd_seq = [(X @ Beta) + np.random.normal(0, 1, n) for X in X_set]
    RD_input     = [(X_set[t], Y_lin_rd_seq[t]) for t in range(T)]

    # ----- Logistic: use own (X_t, Y_t) for each round -----
    eta_log_seq = [X @ Beta for X in X_set]
    probs_seq   = [expit(eta) for eta in eta_log_seq]
    Y_log_seq   = [np.random.binomial(1, probs) for probs in probs_seq]
    LOG_input   = [(X_set[t], Y_log_seq[t]) for t in range(T)]

    # ----- Poisson: use own (X_t, Y_t) for each round -----
    eta_poi_seq = [np.clip(X @ Beta, -20.0, 20.0) for X in X_set]
    mu_seq      = [np.exp(eta) for eta in eta_poi_seq]
    Y_poi_seq   = [np.random.poisson(mu) for mu in mu_seq]
    POI_input   = [(X_set[t], Y_poi_seq[t]) for t in range(T)]

    # Select a fixed design matrix X_fixed (e.g., use the X from round 0)
    X_fixed = X_set[0]
    # For each round, the true response for Fresh is generated based on the same X_fixed (only noise changes)
    Y_lin_fd_seq = [(X_fixed @ Beta) + np.random.normal(0, 1, n) for _ in range(T)]
    # Note: For Fresh_Data.LR, D_0 is a list of length T, each element is (X_fixed, Y_t)
    LR_input_fd  = [(X_fixed, Y_lin_fd_seq[t]) for t in range(T)]

    # ====== One-dimensional fresh sequence required for CDF estimation (consistent with Fully Synthetic approach)======
    D_cdf_set = [np.random.normal(0.0, 1.0, n) for _ in range(T)]

    # ====== Run Fresh_Data ======
    fd = DFD.Fresh_Data(T, w, 1)

    # Multivariate Gaussian (keep original)
    fd.Gaussian_esti(X_set, M, S, seed=rep, start=Start)

    # Random-design LR (use own (X_t, Y_t) for each round)
    fd.RandomLR(RD_input, Beta, seed=rep, start=Start, noise_std=1.0)

    # Logistic (use own (X_t, Y_t) for each round)
    fd.Logistic(LOG_input, Beta, seed=rep, start=Start, max_iter=200)

    # Poisson (use own (X_t, Y_t) for each round)
    fd.Poisson(POI_input, Beta, seed=rep, start=Start, max_iter=300)

    # CDF (nonparametric fresh version)
    fd.CDF_Esti(D_cdf_set, seed=rep, start=Start)

    # ====== Collect results ======
    res = [
        getattr(fd, "Final_loss_mean", np.nan),
        getattr(fd, "Final_loss_var",  np.nan),
        getattr(fd, "Final_loss_beta_random", np.nan),       # random-design
        getattr(fd, "Final_loss_logistic",    np.nan),
        getattr(fd, "Final_loss_poisson",     np.nan),
        getattr(fd, "Final_loss_cdf",         np.nan),       # <- Added: CDF
        w, rep
    ]
    return res


# ========================
# One plot per model (style consistent with Scenario I)
# ========================
def plot_s2_results_each(csv_file, out_dir):
    df = pd.read_csv(csv_file)

    # Take the mean for all reps by w
    df_mean = df.groupby("w").mean(numeric_only=True).reset_index()

    models = {
        "Final_E_Mean":         ("Gaussian Mean", "S2_Mean.png", "red"),
        "Final_E_Var":          ("Gaussian Variance", "S2_Var.png", "blue"),
        "Final_E_beta_LR":  ("Linear Regression", "S2_LR.png", "green"),
        "Final_E_logistic":     ("Logistic Regression",        "S2_Logistic.png",   "purple"),
        "Final_E_poisson":      ("Poisson Regression",         "S2_Poisson.png",    "orange"),
        "Final_E_CDF":          ("CDF Distance",               "S2_CDF.png",        "brown"),   # <- Added
    }

    for col, (ylabel, filename, color) in models.items():
        if col not in df_mean.columns:
            print(f"[warn] column {col} not in CSV; skip plotting this figure.")
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#ADD8E6')   # Light blue background

        # Plot the curve
        ax.plot(df_mean["w"], df_mean[col], linestyle='-.', marker=None,
                color=color)

        # Mark the minimum point
        min_idx = df_mean[col].idxmin()
        min_w = df_mean.loc[min_idx, "w"]
        min_val = df_mean.loc[min_idx, col]
        ax.scatter(min_w, min_val, color=color, s=60, marker='o',
                   label="Minimal Estimation Error")

        # Labels and title
        ax.set_xlabel("Weighting Parameter: $w$")
        ax.set_ylabel("Estimation Error")
        ax.legend()

        # Save the plot
        out_path = Path(out_dir) / filename
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

# ========================
# Main Program
# ========================
if __name__ == "__main__":
    time1 = time.time()

    # Parallel and batch configuration
    NUM_PROCS  = 10
    CHUNK_SIZE = 200

    # Output directory
    out_dir  = Path("Result_Folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "results_S2.csv"

    # Case combinations
    ws  = np.round(np.arange(0.20, 0.80, 0.02), 2)
    reps = range(2000)
    Cases_set = [(float(w), int(rep)) for w in ws for rep in reps]

    # CSV header (including CDF)
    header = [
        "Final_E_Mean", "Final_E_Var",
        "Final_E_beta_LR", "Final_E_logistic", "Final_E_poisson",
        "Final_E_CDF",          # <- Added
        "w", "rep"
    ]
    if not out_csv.exists():
        pd.DataFrame(columns=header).to_csv(out_csv, index=False)

    # Parallel processing in chunks & append to disk
    with Pool(processes=NUM_PROCS) as pool:
        for i in tqdm(range(0, len(Cases_set), CHUNK_SIZE)):
            batch = Cases_set[i: i + CHUNK_SIZE]
            batch_results = pool.starmap(S2_w, batch)
            df_batch = pd.DataFrame(batch_results, columns=header)
            df_batch.to_csv(out_csv, mode='a', header=False, index=False)

            if (i // CHUNK_SIZE) % 10 == 0:
                print(f"Progress: {i + len(batch):,}/{len(Cases_set):,}")
    plot_s2_results_each("Result_Folder/results_S2.csv",'Result_Folder')



