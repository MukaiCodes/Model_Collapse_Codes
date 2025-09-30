# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# ---------------------------
# 1. Data Cleaning
# ---------------------------
def load_clean_adult(train_path="Data/adult.data", test_path="Data/adult.test"):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    train = pd.read_csv(train_path, header=None, names=columns,
                        na_values=" ?", skipinitialspace=True)
    test = pd.read_csv(test_path, header=None, names=columns,
                       na_values=" ?", skiprows=1, skipinitialspace=True)

    df = pd.concat([train, test], axis=0).dropna().reset_index(drop=True)
    return df

# ---------------------------
# 2. Obtain Real Distribution
# ---------------------------
def get_true_distribution(df, col):
    counts = df[col].value_counts(normalize=True).sort_index()
    categories = counts.index.tolist()
    true_dist = counts.values
    return categories, true_dist

# ---------------------------
# 3. Recursive Training
# ---------------------------
def recursive_distribution(df, col, true_dist, categories, n=500, m=500, T=50, w=0.6, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    # Initialization: Real Data
    real_init = rng.choice(df[col], size=n, replace=True)
    est_counts = pd.Series(real_init).value_counts(normalize=True)
    est_dist = np.array([est_counts.get(cat, 0) for cat in categories])
    mse_set = []
    for t in range(1, T+1):
        # real
        real_sample = rng.choice(df[col], size=n, replace=True)
        real_counts = pd.Series(real_sample).value_counts(normalize=True)
        real_dist = np.array([real_counts.get(cat, 0) for cat in categories])

        # synthetic
        syn_sample = rng.choice(categories, size=m, replace=True, p=est_dist)
        syn_counts = pd.Series(syn_sample).value_counts(normalize=True)
        syn_dist = np.array([syn_counts.get(cat, 0) for cat in categories])

        # weighted update
        est_dist = w * real_dist + (1 - w) * syn_dist

        mse = np.mean((est_dist - true_dist) ** 2)
        mse_set.append(mse)
    return np.array(mse_set)[-20::].mean(), est_dist

# ---------------------------
# 4. （different w）
# ---------------------------
def run_experiment_different_w(df, col, n=500, m=500, T=50, reps=50):
    categories, true_dist = get_true_distribution(df, col)

    golden = (np.sqrt(5) - 1) / 2
    w_list = sorted(set(np.arange(0.2, 0.8, 0.02).tolist()))

    results = []
    for w in tqdm(w_list, desc=f"Running {col}"):
        losses = []
        for rep in range(reps):
            rng = np.random.default_rng(rep)
            mse, _ = recursive_distribution(df, col, true_dist, categories, n=n, m=m, T=T, w=w, rng=rng)
            losses.append(mse)
        avg_mse = np.mean(losses)
        results.append((w, avg_mse))
    return results, golden

# ---------------------------
# 5. plot
# ---------------------------
if __name__ == "__main__":
    df = load_clean_adult()

    discrete_cols = ["workclass", "education", "marital_status"]
    n, m, T, reps = 500, 500, 100, 1000

    out_dir = Path("Result_Folder")
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in discrete_cols:
        results, golden = run_experiment_different_w(df, col, n, m, T, reps)
        ws, losses = zip(*results)

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#ADD8E6')
        plt.plot(ws, losses, marker="o", label="Err$(F_{\infty}(\cdot|w,m=n))$")
        # Minimal Point
        min_idx = int(np.argmin(losses))
        plt.scatter(ws[min_idx], losses[min_idx], color="red", zorder=5,
                    label=f"min at w={ws[min_idx]:.3f}")
        plt.annotate(f"min w={ws[min_idx]:.3f}",
                     (ws[min_idx], losses[min_idx]),
                     textcoords="offset points", xytext=(0,10), ha='center', color="red")
        # Golden Ratio
        plt.axvline(golden, color="green", linestyle="--", label="Golden ratio (~0.618)")
        plt.xlabel("Weight on Real Data (w)")
        plt.ylabel("Estimation Error")
        plt.legend()
        plt.grid(True)

        out_path = out_dir / f"{col}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {out_path}")
        plt.close()
