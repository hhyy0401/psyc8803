#!/usr/bin/env python3
"""
Analysis 1 & 2: Feature Importance + DDM Parameter Prediction
==============================================================
Analysis 1: Correlation (FDR-corrected) between EEG features and DDM params
Analysis 2: Regression with 6 models + feature importance extraction

Input:
    data/resting_subject_features.csv   EEG features (~231)
    data/DDM_Scores.csv                 DDM parameters (a, v, t0)
    output/psd_matrix.npz               ROI-averaged PSD for 1D-CNN (optional)

Output:
    output/correlation/
        correlation_fdr.csv
        correlation_report.md
    output/regression/
        regression_results.csv
        lasso_coef_tracking.csv
        shap_{target}.csv
        regression_report.md

Usage:
    conda activate base
    python code/analysis_features.py
"""

import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(os.environ.get("EEG_BASE_DIR", "/Users/hkim3239/GaTech Dropbox/Hyunju Kim/EEG"))
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"
CORR_DIR = OUT_DIR / "correlation"
REG_DIR = OUT_DIR / "regression"

N_FOLDS = 10
N_PERM = 100
N_JOBS = int(os.environ.get("EEG_N_JOBS", "-1"))
TOP_K_FEATURES = 15  # feature selection: top-k by |Pearson r| within each fold
MIN_VALID_N = 50     # drop features with fewer than this many non-NaN values
RANDOM_STATE = 42

# DL
DL_EPOCHS = 300
DL_LR = 1e-3
DL_WD = 1e-2
DL_PATIENCE = 30
DL_BATCH = 32

# Prior results for report comparison
PRIOR_FDR_SIG = 9  # previous analysis: 9 FDR-sig (all aperiodic, FOOOF 42 features)
PRIOR_BEST_R2 = 0.044  # previous best: Speed_Mid_v, FOOOF+SVR


# =============================================================================
# Data Loading
# =============================================================================
def load_eeg():
    df = pd.read_csv(DATA_DIR / "resting_subject_features.csv")
    if "Subject" in df.columns:
        df = df.rename(columns={"Subject": "subject"})
    meta = ["processing_failed", "error", "qc_flag", "n_bad_channels",
            "n_epochs_total", "n_epochs_kept", "flag_low_epochs"]
    df = df.drop(columns=[c for c in meta if c in df.columns], errors="ignore")
    df = df[df["subject"].notna()]
    # Drop features with too few valid values (FOOOF peaks not detected)
    feat_cols = [c for c in df.columns if c != "subject"]
    low_n = [c for c in feat_cols if df[c].notna().sum() < MIN_VALID_N]
    if low_n:
        print(f"  Dropping {len(low_n)} features with <{MIN_VALID_N} valid values: "
              f"{low_n[:5]}{'...' if len(low_n) > 5 else ''}")
        df = df.drop(columns=low_n)
    return df


def build_targets():
    ddm = pd.read_csv(DATA_DIR / "DDM_Scores.csv")
    targets = {}
    grand = ddm.groupby("Subject")[["a", "v", "t0"]].mean().reset_index()
    for p in ["a", "v", "t0"]:
        targets[f"GrandMean_{p}"] = grand[["Subject", p]].rename(
            columns={"Subject": "subject", p: "target"})
    for lc in ["Load", "NoLoad"]:
        sub = (ddm[ddm["Load_Condition"] == lc]
               .groupby("Subject")[["a", "v", "t0"]].mean().reset_index())
        for p in ["a", "v", "t0"]:
            targets[f"{lc}_{p}"] = sub[["Subject", p]].rename(
                columns={"Subject": "subject", p: "target"})
    for sc in ddm["Speed_Condition"].dropna().unique():
        sub = (ddm[ddm["Speed_Condition"] == sc]
               .groupby("Subject")[["a", "v", "t0"]].mean().reset_index())
        sc_clean = sc.replace(" ", "_")
        for p in ["a", "v", "t0"]:
            targets[f"{sc_clean}_{p}"] = sub[["Subject", p]].rename(
                columns={"Subject": "subject", p: "target"})
    return targets


def load_psd_matrix():
    psd_path = OUT_DIR / "psd_matrix.npz"
    if psd_path.exists():
        data = np.load(psd_path)
        return data["psd"], data["subjects"]
    return None, None


def prepare_Xy(eeg, target_df, feature_cols):
    """Merge EEG + target, impute NaN, return X, y arrays."""
    merged = eeg.merge(target_df, on="subject", how="inner").dropna(subset=["target"])
    if len(merged) < 20:
        return None, None, None
    X = merged[feature_cols].values.astype(np.float64)
    y = merged["target"].values.astype(np.float64)
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_med[j] if np.isfinite(col_med[j]) else 0
    return X, y, merged


# =============================================================================
# Analysis 1: Correlation (FDR)
# =============================================================================
def run_correlation_fdr(eeg, targets):
    feature_cols = [c for c in eeg.columns if c != "subject"]
    results = []
    for target_name, target_df in targets.items():
        merged = eeg.merge(target_df, on="subject", how="inner").dropna(subset=["target"])
        if len(merged) < 20:
            continue
        y = merged["target"].values
        p_vals, r_vals, rho_vals = [], [], []
        for feat in feature_cols:
            x = merged[feat].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                p_vals.append(1.0); r_vals.append(0.0); rho_vals.append(0.0)
                continue
            r, p = pearsonr(x[mask], y[mask])
            rho = spearmanr(x[mask], y[mask]).correlation
            p_vals.append(p); r_vals.append(r); rho_vals.append(rho)
        p_arr = np.array(p_vals)
        valid_mask = np.isfinite(p_arr)
        reject = np.zeros(len(p_arr), dtype=bool)
        q_vals = np.full(len(p_arr), np.nan)
        if valid_mask.sum() > 0:
            rej_v, q_v, _, _ = multipletests(p_arr[valid_mask], method="fdr_bh", alpha=0.05)
            reject[valid_mask] = rej_v
            q_vals[valid_mask] = q_v
        for i, feat in enumerate(feature_cols):
            results.append({
                "target": target_name, "feature": feat,
                "pearson_r": r_vals[i], "spearman_rho": rho_vals[i],
                "p_value": p_vals[i], "fdr_q": q_vals[i],
                "fdr_significant": reject[i], "n": len(merged),
            })
    return pd.DataFrame(results)


def write_correlation_report(corr_df, path):
    """Write interpretation report for correlation analysis."""
    sig = corr_df[corr_df["fdr_significant"]]
    n_sig = len(sig)

    lines = [
        "# Correlation Analysis Report (FDR-corrected)",
        "",
        "## Summary",
        "",
        f"- Total tests: {len(corr_df)}",
        f"- FDR-significant (q < 0.05): **{n_sig}**",
        f"- Previous analysis (42 FOOOF features): {PRIOR_FDR_SIG} FDR-sig",
        "",
    ]

    if n_sig > PRIOR_FDR_SIG:
        lines.append(f"**Improvement**: {n_sig} vs {PRIOR_FDR_SIG} prior — "
                      "expanded feature set (connectivity, entropy) yielded more significant associations.")
    elif n_sig == 0:
        lines.append("**No FDR-significant correlations found.** "
                      "This is consistent with weak individual feature-DDM relationships.")
    lines.append("")

    # Categorize significant features
    if n_sig > 0:
        lines.append("## FDR-Significant Features")
        lines.append("")
        lines.append("| Target | Feature | Category | Pearson r | FDR q |")
        lines.append("|--------|---------|----------|-----------|-------|")
        for _, row in sig.sort_values("fdr_q").iterrows():
            feat = row["feature"]
            if "conn_" in feat:
                cat = "Connectivity"
            elif "entropy" in feat:
                cat = "Entropy"
            elif "aperiodic" in feat:
                cat = "Aperiodic"
            elif "peak_" in feat:
                cat = "Periodic (FOOOF)"
            elif "asym_" in feat:
                cat = "Asymmetry"
            else:
                cat = "Band Power"
            lines.append(
                f"| {row['target']} | {feat} | {cat} | "
                f"{row['pearson_r']:+.3f} | {row['fdr_q']:.4f} |")
        lines.append("")

        # Category breakdown
        cats = {}
        for _, row in sig.iterrows():
            feat = row["feature"]
            if "conn_" in feat: c = "Connectivity"
            elif "entropy" in feat: c = "Entropy"
            elif "aperiodic" in feat: c = "Aperiodic"
            elif "peak_" in feat: c = "Periodic"
            elif "asym_" in feat: c = "Asymmetry"
            else: c = "Band Power"
            cats[c] = cats.get(c, 0) + 1

        lines.append("### By Feature Category")
        lines.append("")
        for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {cnt} FDR-significant")
        lines.append("")

    # Literature alignment
    lines += [
        "## Alignment with Prior Literature",
        "",
        "| Hypothesis | Prior Evidence | Current Result |",
        "|-----------|---------------|----------------|",
    ]

    # Check specific hypotheses
    aperiodic_v = sig[(sig["feature"].str.contains("aperiodic_exponent")) &
                       (sig["target"].str.endswith("_v"))] if n_sig > 0 else pd.DataFrame()
    aperiodic_a = sig[(sig["feature"].str.contains("aperiodic_exponent")) &
                       (sig["target"].str.endswith("_a"))] if n_sig > 0 else pd.DataFrame()
    conn_sig = sig[sig["feature"].str.contains("conn_")] if n_sig > 0 else pd.DataFrame()
    entropy_sig = sig[sig["feature"].str.contains("entropy")] if n_sig > 0 else pd.DataFrame()

    lines.append(
        f"| Aperiodic exponent → v | ★★★ (Euler 2024, Pathania 2022) | "
        f"{'Confirmed (' + str(len(aperiodic_v)) + ' sig)' if len(aperiodic_v) > 0 else 'Not significant'} |")
    lines.append(
        f"| Aperiodic exponent → a | Novel (prior: r=-0.23~-0.26) | "
        f"{'Confirmed (' + str(len(aperiodic_a)) + ' sig)' if len(aperiodic_a) > 0 else 'Not significant'} |")
    lines.append(
        f"| Connectivity → DDM | Theoretical, untested | "
        f"{'**NEW: ' + str(len(conn_sig)) + ' sig**' if len(conn_sig) > 0 else 'Not significant'} |")
    lines.append(
        f"| Entropy → DDM | Emerging | "
        f"{'**NEW: ' + str(len(entropy_sig)) + ' sig**' if len(entropy_sig) > 0 else 'Not significant'} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Analysis 2: Regression
# =============================================================================

# ---- ML Models ----
def get_ml_models():
    return {
        "Ridge": {
            "model": RidgeCV(alphas=np.logspace(-3, 3, 50)), "grid": None},
        "Lasso": {
            "model": LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=10000),
            "grid": None},
        "ElasticNet": {
            "model": ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                  alphas=np.logspace(-3, 1, 30), cv=5, max_iter=10000),
            "grid": None},
        "XGBoost": {
            "model": XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0),
            "grid": {"max_depth": [2, 3], "learning_rate": [0.05, 0.1],
                     "reg_alpha": [0, 0.1], "reg_lambda": [1, 5]}},
    }


# ---- DL Models ----
class RegMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n_ch, n_freq):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))
    def forward(self, x):
        return self.fc(self.conv(x).squeeze(-1)).squeeze(-1)


def train_dl(model, X_train, y_train, val_frac=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=DL_WD)
    loss_fn = nn.MSELoss()
    # Hold out validation set for early stopping
    n = len(X_train)
    n_val = max(int(n * val_frac), 1)
    perm = np.random.RandomState(RANDOM_STATE).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_t = torch.FloatTensor(X_train[tr_idx]).to(device)
    y_t = torch.FloatTensor(y_train[tr_idx]).to(device)
    X_v = torch.FloatTensor(X_train[val_idx]).to(device)
    y_v = torch.FloatTensor(y_train[val_idx]).to(device)
    best_loss, patience_cnt, best_state = float("inf"), 0, None
    for _ in range(DL_EPOCHS):
        model.train()
        if len(X_t) > DL_BATCH:
            idx = torch.randperm(len(X_t), device=device)
            for s in range(0, len(X_t), DL_BATCH):
                batch = idx[s: s + DL_BATCH]
                opt.zero_grad()
                loss_fn(model(X_t[batch]), y_t[batch]).backward()
                opt.step()
        else:
            opt.zero_grad()
            loss_fn(model(X_t), y_t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            cur = loss_fn(model(X_v), y_v).item()
        if cur < best_loss - 1e-5:
            best_loss = cur
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= DL_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_dl(model, X):
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy()


# ---- CV ----
def select_top_k(X_tr, y_tr, k=TOP_K_FEATURES):
    """Select top-k features by |Pearson r| within fold (vectorized)."""
    n_feat = X_tr.shape[1]
    if n_feat <= k:
        return np.arange(n_feat)
    # Vectorized correlation: r = cov(X,y) / (std(X) * std(y))
    X_c = X_tr - X_tr.mean(axis=0, keepdims=True)
    y_c = y_tr - y_tr.mean()
    std_x = np.sqrt((X_c ** 2).sum(axis=0))
    std_y = np.sqrt((y_c ** 2).sum())
    std_x[std_x < 1e-10] = np.inf  # zero-variance → correlation 0
    corrs = np.abs(X_c.T @ y_c) / (std_x * std_y)
    return np.argsort(corrs)[-k:]


def run_cv_ml(X, y, model_cfg, feature_names):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full(len(y), np.nan)
    coef_records = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        # Feature selection within fold (prevents leakage)
        sel_idx = select_top_k(X_tr, y_tr)
        X_tr_sel = X_tr[:, sel_idx]
        X_te_sel = X_te[:, sel_idx]
        sel_names = [feature_names[i] for i in sel_idx]
        model = clone(model_cfg["model"])
        grid = model_cfg["grid"]
        if grid:
            gs = GridSearchCV(model, grid, cv=min(3, len(y_tr)), scoring="r2", n_jobs=1)
            gs.fit(X_tr_sel, y_tr)
            best = gs.best_estimator_
            preds[test_idx] = best.predict(X_te_sel)
        else:
            model.fit(X_tr_sel, y_tr)
            best = model
            preds[test_idx] = model.predict(X_te_sel)
        if hasattr(best, "coef_"):
            coefs = best.coef_.flatten()
            for i, fname in enumerate(sel_names):
                if i < len(coefs):
                    coef_records.append({
                        "fold": fold, "feature": fname,
                        "coefficient": coefs[i],
                        "nonzero": abs(coefs[i]) > 1e-10})
    return preds, coef_records


def run_cv_dl(X, y, model_fn):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr = y[train_idx]
        shape_tr, shape_te = X_tr.shape, X_te.shape
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(shape_tr)
        X_te = sc.transform(X_te.reshape(len(X_te), -1)).reshape(shape_te)
        model = model_fn(X_tr.shape)
        model = train_dl(model, X_tr, y_tr)
        preds[test_idx] = predict_dl(model, X_te)
    return preds


# ---- Metrics ----
def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return {"r2": np.nan, "spearman": np.nan, "pearson": np.nan, "mae": np.nan}
    return {
        "r2": r2_score(yt, yp), "spearman": spearmanr(yt, yp).correlation,
        "pearson": pearsonr(yt, yp)[0], "mae": mean_absolute_error(yt, yp)}


# ---- Permutation (lightweight) ----
# For permutation tests, skip GridSearchCV and use fixed params for speed.
_PERM_MODELS = {
    "Ridge":      lambda: RidgeCV(alphas=np.logspace(-3, 3, 50)),
    "Lasso":      lambda: LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=10000),
    "ElasticNet": lambda: ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000),
    "XGBoost":    lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0),
}


def _perm_one(seed, X, y, model_name):
    """Fast permutation: fixed model (no GridSearchCV), with feature selection."""
    rng = np.random.RandomState(seed)
    y_perm = rng.permutation(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        sel = select_top_k(X_tr, y_perm[train_idx])
        model = _PERM_MODELS[model_name]()
        model.fit(X_tr[:, sel], y_perm[train_idx])
        preds[test_idx] = model.predict(X_te[:, sel])
    return r2_score(y, preds)


def permutation_pvalue(X, y, model_name, actual_r2):
    if actual_r2 < -0.1:
        return 1.0
    perm_r2 = [_perm_one(i, X, y, model_name) for i in range(N_PERM)]
    return (np.sum(np.array(perm_r2) >= actual_r2) + 1) / (N_PERM + 1)


# =============================================================================
# Dimensionality Reduction Models (--dim-reduce)
# =============================================================================
DR_N_COMPONENTS = [5, 10, 15, 20, 30]
DR_INNER_FOLDS = 3


def _inner_cv_best_k(X_tr, y_tr, model_class, k_list, **model_kw):
    """Inner CV to select best n_components."""
    inner_kf = KFold(n_splits=DR_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_score, best_k = -np.inf, k_list[0]
    max_k = min(X_tr.shape[0] - X_tr.shape[0] // DR_INNER_FOLDS, X_tr.shape[1]) - 1
    for k in k_list:
        if k >= max_k:
            continue
        scores = []
        for i_tr, i_val in inner_kf.split(X_tr):
            try:
                m = model_class(n_components=k, **model_kw)
                if hasattr(m, 'fit_transform'):
                    X_proj_tr = m.fit_transform(X_tr[i_tr], y_tr[i_tr] if model_class == PLSRegression else None)
                    X_proj_val = m.transform(X_tr[i_val])
                    if model_class == PLSRegression:
                        pred = m.predict(X_tr[i_val]).ravel()
                    else:
                        ridge = RidgeCV(alphas=np.logspace(-3, 3, 30))
                        if isinstance(X_proj_tr, tuple):
                            X_proj_tr = X_proj_tr[0]
                        ridge.fit(X_proj_tr, y_tr[i_tr])
                        if isinstance(X_proj_val, tuple):
                            X_proj_val = X_proj_val[0]
                        pred = ridge.predict(X_proj_val)
                    scores.append(r2_score(y_tr[i_val], pred))
            except Exception:
                continue
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_k = k
    return best_k


def run_cv_pca_ridge(X, y):
    """PCA + Ridge with inner CV for n_components."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full(len(y), np.nan)
    chosen_k = []
    for train_idx, test_idx in kf.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        k = _inner_cv_best_k(X_tr, y[train_idx], PCA, DR_N_COMPONENTS)
        pca = PCA(n_components=k)
        X_tr_p = pca.fit_transform(X_tr)
        X_te_p = pca.transform(X_te)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
        ridge.fit(X_tr_p, y[train_idx])
        preds[test_idx] = ridge.predict(X_te_p)
        chosen_k.append(k)
    return preds, chosen_k


def run_cv_sparse_pca_ridge(X, y, n_components=15):
    """Sparse PCA + Ridge with fixed k (no inner CV for speed)."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        spca = SparsePCA(n_components=n_components, alpha=1.0,
                         random_state=RANDOM_STATE, max_iter=200)
        X_tr_p = spca.fit_transform(X_tr)
        X_te_p = spca.transform(X_te)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
        ridge.fit(X_tr_p, y[train_idx])
        preds[test_idx] = ridge.predict(X_te_p)
    return preds, [n_components] * N_FOLDS


def run_cv_pls(X, y):
    """PLS with inner CV for n_components."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.full(len(y), np.nan)
    chosen_k = []
    k_list = [2, 5, 10, 15, 20]
    for train_idx, test_idx in kf.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        k = _inner_cv_best_k(X_tr, y[train_idx], PLSRegression, k_list, scale=False)
        pls = PLSRegression(n_components=k, scale=False)
        pls.fit(X_tr, y[train_idx])
        preds[test_idx] = pls.predict(X_te).ravel()
        chosen_k.append(k)
    return preds, chosen_k


# ---- SHAP ----
def compute_shap(X, y, feature_names):
    try:
        import shap
    except ImportError:
        return None
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                         random_state=RANDOM_STATE, verbosity=0)
    model.fit(X_sc, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sc)
    return pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)


# ---- Regression Report ----
def write_regression_report(reg_df, coef_df, shap_results, path):
    lines = [
        "# Regression Analysis Report",
        "",
        "## Summary",
        "",
        f"- Models: {', '.join(reg_df['model'].unique())}",
        f"- Targets: {reg_df['target'].nunique()}",
        f"- CV: {N_FOLDS}-fold, permutation test ({N_PERM} iter, ML only)",
        f"- Previous best R²: {PRIOR_BEST_R2:.3f} (FOOOF 42 features)",
        "",
    ]

    # Best per target
    lines += ["## Best Model per Target", "",
              "| Target | Model | CV R² | Spearman ρ | Perm p | vs Prior |",
              "|--------|-------|-------|-----------|--------|----------|"]
    for target in sorted(reg_df["target"].unique()):
        sub = reg_df[reg_df["target"] == target]
        best = sub.loc[sub["r2"].idxmax()]
        p_str = f"{best['perm_p']:.3f}" if np.isfinite(best["perm_p"]) else "—"
        sig = "**" if best["perm_p"] < 0.05 else "*" if best["perm_p"] < 0.1 else ""
        delta = best["r2"] - PRIOR_BEST_R2
        vs = f"{delta:+.4f}" if not np.isnan(best["r2"]) else "—"
        lines.append(
            f"| {target} | {best['model']} | {best['r2']:+.4f} | "
            f"{best['spearman']:+.4f} | {p_str}{sig} | {vs} |")
    lines.append("")

    # Best per DDM parameter
    lines += ["## Best per DDM Parameter", ""]
    for param in ["a", "v", "t0"]:
        sub = reg_df[reg_df["target"].str.endswith(f"_{param}")]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["r2"].idxmax()]
        p_str = f"{best['perm_p']:.3f}" if np.isfinite(best["perm_p"]) else "—"
        lines.append(
            f"- **{param}**: R²={best['r2']:+.4f} ({best['model']} on {best['target']}, p={p_str})")
    lines.append("")

    # Lasso tracking
    if coef_df is not None and len(coef_df) > 0:
        lasso_coef = coef_df[coef_df["model"] == "Lasso"]
        if len(lasso_coef) > 0:
            lines += ["## Lasso Feature Selection", "",
                       "Features consistently selected (non-zero) across CV folds:", ""]
            for target in sorted(lasso_coef["target"].unique()):
                if not target.startswith("GrandMean"):
                    continue
                sub = lasso_coef[lasso_coef["target"] == target]
                freq = (sub.groupby("feature")["nonzero"].sum()
                        .sort_values(ascending=False))
                top = freq[freq > 0].head(10)
                if len(top) == 0:
                    continue
                lines.append(f"**{target}**:")
                for feat, cnt in top.items():
                    robust = " ← robust" if cnt >= 7 else ""
                    lines.append(f"- {feat}: {int(cnt)}/{N_FOLDS}{robust}")
                lines.append("")

    # SHAP
    if shap_results:
        lines += ["## SHAP Feature Importance (XGBoost)", ""]
        for target, imp_df in shap_results.items():
            if imp_df is None:
                continue
            lines.append(f"**{target}** — top 10:")
            for _, row in imp_df.head(10).iterrows():
                lines.append(f"- {row['feature']}: {row['mean_abs_shap']:.4f}")
            lines.append("")

    # Interpretation
    lines += [
        "## Interpretation",
        "",
        "### Comparison with Prior Analysis",
        "",
    ]
    best_overall = reg_df.loc[reg_df["r2"].idxmax()]
    if best_overall["r2"] > PRIOR_BEST_R2:
        lines.append(
            f"Best R² improved from {PRIOR_BEST_R2:.3f} to {best_overall['r2']:.4f} "
            f"({best_overall['model']} on {best_overall['target']}). "
            "The expanded feature set (connectivity + entropy) contributed to this improvement.")
    else:
        lines.append(
            f"Best R² ({best_overall['r2']:.4f}) did not substantially exceed prior ({PRIOR_BEST_R2:.3f}). "
            "Resting-state EEG has limited individual-level predictive power for DDM parameters, "
            "consistent with the weak signal hypothesis.")
    lines.append("")

    lines += [
        "### Key Findings",
        "",
        "1. **Feature importance convergence**: Features appearing in both FDR-significant "
        "correlations AND Lasso/SHAP rankings provide the strongest evidence for EEG-DDM relationships.",
        "",
        "2. **Aperiodic exponent**: Prior analysis identified this as the only robust predictor. "
        "Current results with expanded features should be compared.",
        "",
        "3. **Connectivity features**: If conn_* features appear in top rankings, this suggests "
        "network-level dynamics contribute beyond single-ROI power (novel finding).",
        "",
        "4. **Entropy features**: If entropy features are selected, this indicates temporal "
        "complexity of resting EEG relates to decision-making efficiency.",
        "",
        "### Literature Alignment",
        "",
        "- Aperiodic exponent → drift rate (v): Consistent with Euler et al. (2024), "
        "Pathania et al. (2022) — steeper 1/f slope reflects better E/I balance → faster evidence accumulation.",
        "- IAF → processing speed: Grandy (2013), Finley (2024) — higher alpha frequency → faster temporal resolution.",
        "- TBR → boundary separation: Mixed evidence in prior analysis (predicted v instead of a).",
        "- Temporal ROI dominance: Unexpected finding from prior analysis — warrants investigation for muscle artifact.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================
def main(args):
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    REG_DIR.mkdir(parents=True, exist_ok=True)
    t0_global = time.time()

    if args.dim_reduce:
        print(">> Dimensionality reduction models ENABLED (PCA, PLS, SparsePCA)")

    # ---- Load ----
    print("Loading data...")
    eeg = load_eeg()
    targets = build_targets()
    psd_matrix, psd_subjects = load_psd_matrix()
    feature_cols = [c for c in eeg.columns if c != "subject"]
    print(f"  EEG: {len(eeg)} subjects × {len(feature_cols)} features")
    print(f"  Targets: {len(targets)}")
    if psd_matrix is not None:
        psd_subj_map = {int(s): i for i, s in enumerate(psd_subjects)}
        print(f"  PSD: {psd_matrix.shape}")
    else:
        psd_subj_map = {}
        print("  PSD: not found (CNN skipped)")

    HAS_TABNET = False
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor  # noqa: F401
        HAS_TABNET = True
        print("  TabNet: available")
    except ImportError:
        print("  TabNet: not installed (skipping)")

    # ==================================================================
    # Analysis 1: Correlation
    # ==================================================================
    print(f"\n{'='*60}\nAnalysis 1: Correlation (FDR)\n{'='*60}")

    corr_df = run_correlation_fdr(eeg, targets)
    corr_df.to_csv(CORR_DIR / "correlation_fdr.csv", index=False)
    write_correlation_report(corr_df, CORR_DIR / "correlation_report.md")

    n_sig = corr_df["fdr_significant"].sum()
    print(f"  {len(corr_df)} tests, {n_sig} FDR-significant (prior: {PRIOR_FDR_SIG})")
    if n_sig > 0:
        for _, row in corr_df[corr_df["fdr_significant"]].sort_values("fdr_q").head(5).iterrows():
            print(f"    {row['feature']} ↔ {row['target']}: r={row['pearson_r']:+.3f}, q={row['fdr_q']:.4f}")

    # ==================================================================
    # Analysis 2: Regression
    # ==================================================================
    print(f"\n{'='*60}\nAnalysis 2: Regression (6 models)\n{'='*60}")

    ml_models = get_ml_models()
    all_results, all_coefs = [], []
    shap_results = {}

    for ti, (target_name, target_df) in enumerate(targets.items()):
        print(f"\n[{ti+1}/{len(targets)}] {target_name}")
        X, y, merged = prepare_Xy(eeg, target_df, feature_cols)
        if X is None:
            print(f"  Skipping: <20 subjects")
            continue
        print(f"  N={len(y)}, features={X.shape[1]}")

        # ML: run all models first, then permutation for the best
        ml_metrics = {}
        for mname, mcfg in ml_models.items():
            t0 = time.time()
            preds, coefs = run_cv_ml(X, y, mcfg, feature_cols)
            m = compute_metrics(y, preds)
            ml_metrics[mname] = m
            for c in coefs:
                c["model"] = mname; c["target"] = target_name
            all_coefs.extend(coefs)
            print(f"  {mname:12s} R²={m['r2']:+.4f} ρ={m['spearman']:+.4f} ({time.time()-t0:.1f}s)")
            all_results.append({"target": target_name, "model": mname, **m, "perm_p": np.nan, "n": len(y)})

        # Permutation test only for the best ML model per target
        best_ml = max(ml_metrics, key=lambda k: ml_metrics[k]["r2"])
        t0 = time.time()
        pp = permutation_pvalue(X, y, best_ml, ml_metrics[best_ml]["r2"])
        sig = "*" if pp < 0.05 else ""
        print(f"  Perm({best_ml}): p={pp:.3f}{sig} ({time.time()-t0:.1f}s)")
        # Update the result for the best model with perm_p
        for r in all_results:
            if r["target"] == target_name and r["model"] == best_ml:
                r["perm_p"] = pp

        # SHAP (grand mean only)
        if target_name.startswith("GrandMean"):
            print(f"  SHAP...")
            shap_imp = compute_shap(X, y, feature_cols)
            if shap_imp is not None:
                shap_results[target_name] = shap_imp

        # Dimensionality reduction models (if enabled)
        if args.dim_reduce:
            # PCA + Ridge
            t0 = time.time()
            preds_pca, ks_pca = run_cv_pca_ridge(X, y)
            m_pca = compute_metrics(y, preds_pca)
            k_str = f"k={int(np.median(ks_pca))}"
            print(f"  {'PCA+Ridge':12s} R²={m_pca['r2']:+.4f} ρ={m_pca['spearman']:+.4f} ({k_str}, {time.time()-t0:.1f}s)")
            all_results.append({"target": target_name, "model": "PCA+Ridge", **m_pca, "perm_p": np.nan, "n": len(y)})

            # PLS
            t0 = time.time()
            preds_pls, ks_pls = run_cv_pls(X, y)
            m_pls = compute_metrics(y, preds_pls)
            k_str = f"k={int(np.median(ks_pls))}"
            print(f"  {'PLS':12s} R²={m_pls['r2']:+.4f} ρ={m_pls['spearman']:+.4f} ({k_str}, {time.time()-t0:.1f}s)")
            all_results.append({"target": target_name, "model": "PLS", **m_pls, "perm_p": np.nan, "n": len(y)})

            # Sparse PCA + Ridge
            t0 = time.time()
            preds_spca, ks_spca = run_cv_sparse_pca_ridge(X, y)
            m_spca = compute_metrics(y, preds_spca)
            k_str = f"k={int(np.median(ks_spca))}"
            print(f"  {'SparsePCA+R':12s} R²={m_spca['r2']:+.4f} ρ={m_spca['spearman']:+.4f} ({k_str}, {time.time()-t0:.1f}s)")
            all_results.append({"target": target_name, "model": "SparsePCA+Ridge", **m_spca, "perm_p": np.nan, "n": len(y)})

        # MLP
        t0 = time.time()
        preds_mlp = run_cv_dl(X, y, lambda shape: RegMLP(shape[1]))
        m_mlp = compute_metrics(y, preds_mlp)
        print(f"  {'MLP':12s} R²={m_mlp['r2']:+.4f} ρ={m_mlp['spearman']:+.4f} ({time.time()-t0:.1f}s)")
        all_results.append({"target": target_name, "model": "MLP", **m_mlp, "perm_p": np.nan, "n": len(y)})

        # 1D-CNN
        if merged is not None and psd_matrix is not None:
            subj_ids = merged["subject"].values
            psd_idx = [psd_subj_map.get(int(s), -1) for s in subj_ids]
            if all(i >= 0 for i in psd_idx):
                X_psd = psd_matrix[psd_idx]
                t0 = time.time()
                preds_cnn = run_cv_dl(X_psd, y, lambda shape: CNN1D(shape[1], shape[2]))
                m_cnn = compute_metrics(y, preds_cnn)
                print(f"  {'1D-CNN':12s} R²={m_cnn['r2']:+.4f} ρ={m_cnn['spearman']:+.4f} ({time.time()-t0:.1f}s)")
                all_results.append({"target": target_name, "model": "1D-CNN", **m_cnn, "perm_p": np.nan, "n": len(y)})

        # TabNet
        if HAS_TABNET:
            from pytorch_tabnet.tab_model import TabNetRegressor
            t0 = time.time()
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            preds_tab = np.full(len(y), np.nan)
            for train_idx, test_idx in kf.split(X):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X[train_idx]); X_te = sc.transform(X[test_idx])
                tab = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.5,
                                      lambda_sparse=1e-3, verbose=0, seed=RANDOM_STATE)
                tab.fit(X_tr, y[train_idx].reshape(-1, 1),
                        max_epochs=200, patience=20, batch_size=DL_BATCH)
                preds_tab[test_idx] = tab.predict(X_te).flatten()
            m_tab = compute_metrics(y, preds_tab)
            print(f"  {'TabNet':12s} R²={m_tab['r2']:+.4f} ρ={m_tab['spearman']:+.4f} ({time.time()-t0:.1f}s)")
            all_results.append({"target": target_name, "model": "TabNet", **m_tab, "perm_p": np.nan, "n": len(y)})

    # ---- Save ----
    reg_df = pd.DataFrame(all_results)
    reg_df.to_csv(REG_DIR / "regression_results.csv", index=False)

    coef_df = pd.DataFrame(all_coefs) if all_coefs else None
    if coef_df is not None:
        coef_df.to_csv(REG_DIR / "lasso_coef_tracking.csv", index=False)

    for target, imp_df in shap_results.items():
        if imp_df is not None:
            imp_df.to_csv(REG_DIR / f"shap_{target}.csv", index=False)

    write_regression_report(reg_df, coef_df, shap_results, REG_DIR / "regression_report.md")

    print(f"\n{'='*60}")
    print(f"Done in {(time.time()-t0_global)/60:.1f} min")
    print(f"Correlation: {CORR_DIR}")
    print(f"Regression:  {REG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim-reduce", action="store_true",
                        help="Enable PCA, Sparse PCA, PLS regression models")
    args = parser.parse_args()
    main(args)
