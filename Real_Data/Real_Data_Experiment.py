import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ---------------------------
# 1. Data Loading
# ---------------------------
def load_clean_adult(train_path="Data/adult.data", test_path="Data/adult.test"):
    """
    Load and clean the UCI Adult dataset.
    Automatically drops rows with missing values (" ?").
    """
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(
        train_path, header=None, names=columns,
        na_values=" ?", skipinitialspace=True
    )
    test = pd.read_csv(
        test_path, header=None, names=columns,
        na_values=" ?", skiprows=1, skipinitialspace=True
    )

    train_clean = train.dropna().reset_index(drop=True)
    test_clean = test.dropna().reset_index(drop=True)

    return train_clean, test_clean


# ---------------------------
# 2. Data Preparation
# ---------------------------
def prepare_full_data(train_df, test_df):
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    full_df["income"] = (full_df["income"].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)

    features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    X = full_df[features].values
    y = full_df["income"].values
    return X, y, features


def fit_logistic(X, y, sample_weight=None):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y, sample_weight=sample_weight)
    return model


# ---------------------------
# 3. Recursive Training Experiments
# ---------------------------
def recursive_training(X_all, y_all, beta_true, T=50, n=1000, m=1000, w=0.6, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)  # Fix random seed

    # Step 0: initialize model
    idx = rng.choice(len(X_all), size=n, replace=True)
    X_init, y_init = X_all[idx], y_all[idx]
    model = fit_logistic(X_init, y_init)
    loss_set = []
    for t in range(1, T + 1):
        # 1. Sample fresh real data
        idx_real = rng.choice(len(X_all), size=n, replace=True)
        X_real, y_real = X_all[idx_real], y_all[idx_real]

        # 2. Generate synthetic data
        idx_syn = rng.choice(len(X_all), size=m, replace=True)
        X_syn = X_all[idx_syn]
        y_syn_prob = model.predict_proba(X_syn)[:, 1]
        y_syn = rng.binomial(1, y_syn_prob)

        # 3. Perform weighted training
        X_mix = np.vstack([X_real, X_syn])
        y_mix = np.concatenate([y_real, y_syn])

        real_weight = w * (n + m) / n
        syn_weight = (1 - w) * (n + m) / m
        sample_weight = np.concatenate([np.full(n, real_weight), np.full(m, syn_weight)])

        model = fit_logistic(X_mix, y_mix, sample_weight=sample_weight)

        beta_hat = np.concatenate([model.intercept_, model.coef_.ravel()])
        loss = np.linalg.norm(beta_hat - beta_true) ** 2
        loss_set.append(loss)
    return np.mean(np.array(loss_set)[-20::]), beta_hat


# ---------------------------
# 4. Main Program
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    train_df, test_df = load_clean_adult()
    X_all, y_all, features = prepare_full_data(train_df, test_df)

    print("Full dataset size:", X_all.shape)

    # Step 0: pseudo-truth β⋆
    model_true = fit_logistic(X_all, y_all)
    beta_true = np.concatenate([model_true.intercept_, model_true.coef_.ravel()])

    # =====================================
    # Experiment 1: Fix n=m=1000, compare different w
    # =====================================
    golden = (np.sqrt(5) - 1) / 2
    ws = sorted(set(np.arange(0.2, 0.8, 0.02).tolist()))
    results = []
    for w in tqdm(ws):
        losses = []
        for rep in range(1000):  # Repeat multiple times and take average
            loss, _ = recursive_training(X_all, y_all, beta_true, T=100, n=500, m=500, w=w,
                                         rng=np.random.default_rng(rep))
            losses.append(loss)
        avg_loss = np.mean(losses)
        results.append((w, avg_loss))

    print("\nExperiment 1: w vs loss (n=m=1000)")
    for w, loss in results:
        print(f"w={w:.4f}, loss={loss:.4f}")

    import time
    from pathlib import Path

    # Visualization
    ws_list, loss_list = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#ADD8E6')  # Light blue background
    plt.plot(ws_list, loss_list, marker="o", label="Err$(\widehat{\mathbf{\\theta}}_{\infty}(w,1)}))$")

    min_idx = int(np.argmin(loss_list))
    plt.scatter(ws_list[min_idx], loss_list[min_idx], color="red", zorder=5,
                label=f"min at w={ws_list[min_idx]:.3f}")
    plt.annotate(f"min w={ws_list[min_idx]:.3f}",
                 (ws_list[min_idx], loss_list[min_idx]),
                 textcoords="offset points", xytext=(0, 10), ha='center', color="red")

    plt.axvline(golden, color="green", linestyle="--", label="Golden ratio (~0.618)")
    plt.xlabel("Weight on real data (w)")
    plt.ylabel("Estimation Error")
    plt.legend()
    plt.grid(True)

    # ===============================
    # Save figure into Result_Folder
    # ===============================
    out_dir = Path("Result_Folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Logistic.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # Show figure if needed
    plt.show()


