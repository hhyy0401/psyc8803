#!/usr/bin/env python3
"""
Task Embedding Experiment
=========================
Compares five task encoding strategies (Tiers 0–4) for EEG → RT prediction.

Tier 0 : Current baseline  [load, log_penalty]                          (2D)
Tier 1 : SDT / EV features [load, log_rp, tau_acc, strategy_ord,
                             load_rp_interaction]                         (5D)
Tier 2 : + DDM-theoretic   Tier1 + [r_net, delta_a_star, rt_direction]  (8D)
Tier 3 : SBERT instruction text → PCA-8                                  (8D)
Tier 4 : Learned nn.Embedding(10,8) warm-started from Tier1             (8D)

Models  : XGBoost, MLP (classical models use tabular task features only,
          neural models additionally test Tier 4 learned embeddings)

Usage:
    python code/task_embedding_experiment.py
    python code/task_embedding_experiment.py --skip-sbert   # skip Tier 3
    python code/task_embedding_experiment.py --fast         # fewer epochs
"""

import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CANDIDATES = [
    Path("."),
    Path("/storage/project/r-nimam6-0/hkim3239/cogegg"),
]

def _find_base():
    env = os.environ.get("EEG_BASE_DIR")
    if env:
        return Path(env)
    for d in _CANDIDATES:
        if (d / "data" / "resting_subject_features.csv").exists():
            return d
    return Path(".")

BASE_DIR   = _find_base()
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "output" / "task_embeddings"

N_FOLDS      = 5
RANDOM_STATE = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DL_EPOCHS   = 100
DL_LR       = 1e-3
DL_PATIENCE = 15
DL_BATCH    = 128

TARGET_NAMES = ["RT_mean", "RT_sd", "ACC"]

# ---------------------------------------------------------------------------
# Instruction texts (from TaskDescription.pdf)
# ---------------------------------------------------------------------------
INSTRUCTION_TEXTS = {
    ("Speed_Max",    "NoLoad"): (
        "To maximize your score, answer quickly, because errors will not be "
        "penalized very much."
    ),
    ("Speed_Max",    "Load"): (
        "To maximize your score, answer quickly, because errors will not be "
        "penalized very much. Also, there will be the additional task of "
        "memorizing letters presented during feedback."
    ),
    ("Speed_Mid",    "NoLoad"): (
        "To maximize your score, try to answer relatively quickly, because "
        "errors will not be penalized very much."
    ),
    ("Speed_Mid",    "Load"): (
        "To maximize your score, try to answer relatively quickly, because "
        "errors will not be penalized very much. Also, there will be the "
        "additional task of memorizing letters presented during feedback."
    ),
    ("Neutral",      "NoLoad"): (
        "Try to earn as many points as possible to maximize your score."
    ),
    ("Neutral",      "Load"): (
        "Try to earn as many points as possible to maximize your score. "
        "Also, there will be the additional task of memorizing letters "
        "presented during feedback."
    ),
    ("Accuracy_Mid", "NoLoad"): (
        "To maximize your score, try to answer relatively carefully, because "
        "errors will be heavily penalized."
    ),
    ("Accuracy_Mid", "Load"): (
        "To maximize your score, try to answer relatively carefully, because "
        "errors will be heavily penalized. Also, there will be the additional "
        "task of memorizing letters presented during feedback."
    ),
    ("Accuracy_Max", "NoLoad"): (
        "To maximize your score, answer carefully, because errors will be "
        "heavily penalized."
    ),
    ("Accuracy_Max", "Load"): (
        "To maximize your score, answer carefully, because errors will be "
        "heavily penalized. Also, there will be the additional task of "
        "memorizing letters presented during feedback."
    ),
}

CONDITION_INDEX = {
    ("Speed_Max",    "NoLoad"): 0,
    ("Speed_Max",    "Load"):   1,
    ("Speed_Mid",    "NoLoad"): 2,
    ("Speed_Mid",    "Load"):   3,
    ("Neutral",      "NoLoad"): 4,
    ("Neutral",      "Load"):   5,
    ("Accuracy_Mid", "NoLoad"): 6,
    ("Accuracy_Mid", "Load"):   7,
    ("Accuracy_Max", "NoLoad"): 8,
    ("Accuracy_Max", "Load"):   9,
}

# Ordered condition labels for plotting
COND_LABELS = [
    "SpMax_NL", "SpMax_L",
    "SpMid_NL", "SpMid_L",
    "Neut_NL",  "Neut_L",
    "AccMid_NL","AccMid_L",
    "AccMax_NL","AccMax_L",
]

# ---------------------------------------------------------------------------
# Task feature builders
# ---------------------------------------------------------------------------
def tier0_features(load, penalty, reward):
    """Baseline: [load, log_penalty]."""
    return np.array([float(load), np.log(float(penalty))])


def tier1_features(load, penalty, reward):
    """SDT / EV-grounded: 5 features."""
    p, r = float(penalty), float(reward)
    log_rp        = np.log(r / p)
    tau_acc       = p / (r + p)
    ord_map       = {5: 0.00, 10: 0.25, 20: 0.50, 40: 0.75, 80: 1.00}
    strategy_ord  = ord_map.get(int(p), 0.5)
    load_rp_inter = float(load) * log_rp
    return np.array([float(load), log_rp, tau_acc, strategy_ord, load_rp_inter])


def tier2_features(load, penalty, reward):
    """Tier1 + DDM-theoretic: 8 features."""
    t1 = tier1_features(load, penalty, reward)
    p, r = float(penalty), float(reward)
    rp_ratio    = p / r
    r_net       = (r - p) / (r + p)
    delta_a     = np.sqrt((rp_ratio + 1.0) / 2.0) - 1.0
    rt_dir      = float(np.sign(delta_a))
    return np.concatenate([t1, [r_net, delta_a, rt_dir]])


def build_sbert_embeddings(n_components=8):
    """Encode instruction texts with SBERT → PCA-8. Returns dict + pca."""
    from sentence_transformers import SentenceTransformer
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    keys   = list(INSTRUCTION_TEXTS.keys())
    texts  = [INSTRUCTION_TEXTS[k] for k in keys]
    embeds = model.encode(texts, normalize_embeddings=True,
                          show_progress_bar=False)          # (10, 384)
    pca    = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeds)                      # (10, n_components)
    cond_to_embed = {k: reduced[i] for i, k in enumerate(keys)}
    return cond_to_embed, pca


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _impute(X):
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any():
            med = np.nanmedian(X[:, j])
            X[m, j] = med if np.isfinite(med) else 0.0
    return X


def load_data():
    eeg = pd.read_csv(DATA_DIR / "resting_subject_features.csv")
    if "Subject" in eeg.columns:
        eeg = eeg.rename(columns={"Subject": "subject"})
    drop = ["processing_failed", "error", "qc_flag", "n_bad_channels",
            "n_epochs_total", "n_epochs_kept", "flag_low_epochs"]
    eeg  = eeg.drop(columns=[c for c in drop if c in eeg.columns], errors="ignore")
    eeg  = eeg[eeg["subject"].notna()]

    ddm  = pd.read_csv(DATA_DIR / "DDM_Scores.csv")
    keep = ["Subject", "Detailed_Condition", "Penalty", "Reward",
            "Load_Condition", "Speed_Condition",
            "Lexical.ACC", "Lexical.RT_mean", "Lexical.RT_sd"]
    ddm  = ddm[[c for c in keep if c in ddm.columns]].rename(
        columns={"Subject": "subject", "Detailed_Condition": "condition"})

    merged = ddm.merge(eeg, on="subject", how="inner")
    merged = merged.dropna(subset=["Lexical.RT_mean"]).reset_index(drop=True)

    feature_cols = [c for c in eeg.columns if c != "subject"]

    # Add condition index for learned embedding
    merged["cond_idx"] = merged.apply(
        lambda r: CONDITION_INDEX.get(
            (r["Speed_Condition"], r["Load_Condition"]), 0), axis=1)

    return merged, feature_cols


def build_task_arrays(merged, tier, sbert_embeds=None):
    """Build task feature array for a given tier."""
    rows = []
    for _, r in merged.iterrows():
        load    = 1 if r["Load_Condition"] == "Load" else 0
        penalty = float(r["Penalty"])
        reward  = float(r["Reward"])
        if tier == 0:
            rows.append(tier0_features(load, penalty, reward))
        elif tier == 1:
            rows.append(tier1_features(load, penalty, reward))
        elif tier == 2:
            rows.append(tier2_features(load, penalty, reward))
        elif tier == 3:
            if sbert_embeds is None:
                raise ValueError("sbert_embeds required for Tier 3")
            key = (r["Speed_Condition"], r["Load_Condition"])
            rows.append(sbert_embeds.get(key, np.zeros(8)))
        else:
            raise ValueError(f"Unknown tier {tier}")
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    out = {}
    for i, name in enumerate(TARGET_NAMES):
        yt, yp = y_true[:, i], y_pred[:, i]
        mask = np.isfinite(yp) & np.isfinite(yt)
        if mask.sum() < 10:
            out[f"{name}_r2"]  = np.nan
            out[f"{name}_mae"] = np.nan
            out[f"{name}_rho"] = np.nan
        else:
            out[f"{name}_r2"]  = r2_score(yt[mask], yp[mask])
            out[f"{name}_mae"] = mean_absolute_error(yt[mask], yp[mask])
            out[f"{name}_rho"] = spearmanr(yt[mask], yp[mask]).correlation
    return out


# ---------------------------------------------------------------------------
# Classical ML (Ridge / XGBoost)
# ---------------------------------------------------------------------------
def run_xgboost(X, Y, groups):
    gkf  = GroupKFold(n_splits=N_FOLDS)
    preds = np.full_like(Y, np.nan)
    for tr, te in gkf.split(X, groups=groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for oi in range(3):
            m = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=RANDOM_STATE, verbosity=0)
            m.fit(Xtr, Y[tr, oi])
            preds[te, oi] = m.predict(Xte)
    return preds


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class MLPModel(nn.Module):
    def __init__(self, n_in, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )
    def forward(self, x, cond_idx=None):
        return self.net(x)


class MLPWithLearnedEmbedding(nn.Module):
    """MLP that replaces the tabular task-feat slice with a learned embedding."""
    def __init__(self, n_eeg, d_task=8, n_out=3, init_weights=None):
        super().__init__()
        self.n_eeg  = n_eeg
        self.d_task = d_task
        self.task_emb = nn.Embedding(10, d_task)
        if init_weights is not None:
            # warm-start: init_weights is (10, d_src) array; project → d_task
            w = torch.FloatTensor(init_weights)
            proj = nn.Linear(w.shape[1], d_task, bias=False)
            with torch.no_grad():
                projected = proj(w)
                projected = projected - projected.mean(0)  # centre
                self.task_emb.weight.copy_(projected)
        else:
            nn.init.normal_(self.task_emb.weight, std=0.01)
        self.net = nn.Sequential(
            nn.Linear(n_eeg + d_task, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),             nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )
    def forward(self, x_full, cond_idx):
        eeg_part = x_full[:, :self.n_eeg]
        task_emb = self.task_emb(cond_idx)           # (B, d_task)
        return self.net(torch.cat([eeg_part, task_emb], dim=-1))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _train(model, X_t, Y_t, X_v, Y_v,
           cond_t=None, cond_v=None,
           epochs=DL_EPOCHS, lr=DL_LR,
           patience=DL_PATIENCE, batch_size=DL_BATCH):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best_loss, wait, best_state = float("inf"), 0, None

    uses_emb = cond_t is not None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X_t[idx], cond_t[idx]) if uses_emb else model(X_t[idx])
            loss = F.mse_loss(pred, Y_t[idx])
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_v, cond_v) if uses_emb else model(X_v)
            vl = F.mse_loss(vp, Y_v).item()
        if np.isnan(vl):
            vl = float("inf")
        sched.step(vl)
        if vl < best_loss - 1e-5:
            best_loss  = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_mlp(X_eeg, X_task, Y, groups,
            cond_idx=None, epochs=DL_EPOCHS,
            use_learned_emb=False, init_weights=None):
    """
    CV runner for MLP variants.
    - X_eeg      : (N, n_eeg) — always included
    - X_task     : (N, n_task) — tabular task features; ignored when use_learned_emb
    - cond_idx   : (N,) integer condition indices for Tier 4
    - use_learned_emb : if True, ignore X_task and use MLPWithLearnedEmbedding
    - init_weights    : (10, d_src) for warm-starting the embedding table
    """
    gkf   = GroupKFold(n_splits=N_FOLDS)
    preds = np.full_like(Y, np.nan)

    if not use_learned_emb:
        X_full = np.hstack([X_eeg, X_task]) if X_task.shape[1] > 0 else X_eeg
    else:
        X_full = X_eeg   # embedding handles task

    for fold, (tr, te) in enumerate(gkf.split(X_full, groups=groups)):
        sc   = StandardScaler().fit(X_full[tr])
        Xtr  = sc.transform(X_full[tr])
        Xte  = sc.transform(X_full[te])
        y_mu  = Y[tr].mean(0);  y_sig = Y[tr].std(0)
        y_sig[y_sig < 1e-8] = 1.0
        Ytr_n = (Y[tr] - y_mu) / y_sig

        n_val = max(int(len(tr) * 0.15), 1)
        rs    = np.random.RandomState(RANDOM_STATE + fold)
        perm  = rs.permutation(len(tr))
        ti, vi = perm[n_val:], perm[:n_val]

        Xt = torch.FloatTensor(Xtr[ti]).to(DEVICE)
        Yt = torch.FloatTensor(Ytr_n[ti]).to(DEVICE)
        Xv = torch.FloatTensor(Xtr[vi]).to(DEVICE)
        Yv = torch.FloatTensor(Ytr_n[vi]).to(DEVICE)

        ct = cv = None
        if use_learned_emb and cond_idx is not None:
            ct = torch.LongTensor(cond_idx[tr][ti]).to(DEVICE)
            cv = torch.LongTensor(cond_idx[tr][vi]).to(DEVICE)

        if use_learned_emb:
            model = MLPWithLearnedEmbedding(
                Xtr.shape[1], d_task=8,
                init_weights=init_weights).to(DEVICE)
        else:
            model = MLPModel(Xtr.shape[1]).to(DEVICE)

        model = _train(model, Xt, Yt, Xv, Yv,
                       cond_t=ct, cond_v=cv, epochs=epochs)

        Xtest = torch.FloatTensor(Xte).to(DEVICE)
        ct_te = (torch.LongTensor(cond_idx[te]).to(DEVICE)
                 if use_learned_emb and cond_idx is not None else None)
        with torch.no_grad():
            pred_n = model(Xtest, ct_te).cpu().numpy() if use_learned_emb \
                     else model(Xtest).cpu().numpy()
        preds[te] = pred_n * y_sig + y_mu

    return preds


# ---------------------------------------------------------------------------
# Helper: build warm-start weight matrix from Tier-1 features (10 conditions)
# ---------------------------------------------------------------------------
def build_tier1_init_weights():
    """Return (10, 5) array of Tier-1 features for each condition index."""
    penalty_map = {0: 5, 1: 5, 2: 10, 3: 10, 4: 20,
                   5: 20, 6: 40, 7: 40, 8: 80, 9: 80}
    load_map    = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0,
                   5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
    rows = []
    for idx in range(10):
        rows.append(tier1_features(load_map[idx], penalty_map[idx], reward=20))
    return np.array(rows)  # (10, 5)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
TIER_NAMES = {
    "t0_xgb":  "Tier0 (2D)",
    "t1_xgb":  "Tier1 (5D)",
    "t2_xgb":  "Tier2 (8D)",
    "t3_xgb":  "Tier3 SBERT",
    "t0_mlp":  "Tier0 (2D)",
    "t1_mlp":  "Tier1 (5D)",
    "t2_mlp":  "Tier2 (8D)",
    "t3_mlp":  "Tier3 SBERT",
    "t4_mlp":  "Tier4 Learned",
}

MODEL_DISPLAY = {
    "t0_xgb": "XGB-T0", "t1_xgb": "XGB-T1", "t2_xgb": "XGB-T2", "t3_xgb": "XGB-T3",
    "t0_mlp": "MLP-T0", "t1_mlp": "MLP-T1", "t2_mlp": "MLP-T2",
    "t3_mlp": "MLP-T3", "t4_mlp": "MLP-T4",
}

PALETTE = {
    "XGB-T0": "#1f77b4", "XGB-T1": "#aec7e8", "XGB-T2": "#08519c", "XGB-T3": "#6baed6",
    "MLP-T0": "#d62728", "MLP-T1": "#fc8d59", "MLP-T2": "#a50f15",
    "MLP-T3": "#fdae6b", "MLP-T4": "#7b2d8b",
}


def plot_rho_comparison(df, out_path):
    """Bar chart: Spearman rho per model/tier for all 3 targets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    targets    = ["RT_mean", "RT_sd", "ACC"]
    ylabels    = ["Spearman ρ (RT mean)", "Spearman ρ (RT sd)", "Spearman ρ (ACC)"]

    for ax, tgt, ylabel in zip(axes, targets, ylabels):
        col   = f"{tgt}_rho"
        names = df["display_name"].tolist()
        vals  = df[col].tolist()
        colors = [PALETTE.get(n, "#888") for n in names]

        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(tgt.replace("_", " "))
        ax.set_ylim(min(-0.05, min(vals) - 0.05), max(0.55, max(vals) + 0.05))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.03,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Task Embedding Comparison — Spearman ρ by Target", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_from_baseline(df, out_path):
    """Improvement relative to Tier-0 baseline for each model family."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    targets   = ["RT_mean", "RT_sd", "ACC"]

    for ax, tgt in zip(axes, targets):
        col = f"{tgt}_rho"
        baseline_xgb = df.loc[df["key"] == "t0_xgb", col].values
        baseline_mlp = df.loc[df["key"] == "t0_mlp", col].values

        if len(baseline_xgb) == 0 or len(baseline_mlp) == 0:
            continue

        b_xgb = baseline_xgb[0]
        b_mlp = baseline_mlp[0]

        xgb_rows = df[df["key"].str.startswith("t") & df["key"].str.endswith("xgb")
                      & (df["key"] != "t0_xgb")]
        mlp_rows = df[df["key"].str.startswith("t") & df["key"].str.endswith("mlp")
                      & (df["key"] != "t0_mlp")]

        all_rows  = pd.concat([xgb_rows, mlp_rows])
        deltas    = all_rows[col].values - np.where(
            all_rows["key"].str.endswith("xgb"), b_xgb, b_mlp)
        names     = all_rows["display_name"].tolist()
        colors    = [PALETTE.get(n, "#888") for n in names]

        bars = ax.bar(range(len(names)), deltas, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Δρ vs Tier-0 ({tgt})")
        ax.set_ylabel("Δ Spearman ρ")
        for bar, v in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.005 if v >= 0 else v - 0.02,
                    f"{v:+.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Task Embedding Gain over Tier-0 Baseline", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sbert_pca(cond_embed_dict, pca_obj, out_path):
    """2-D scatter of SBERT PCA condition embeddings (PC1 vs PC2)."""
    conds = list(cond_embed_dict.keys())
    pts   = np.array([cond_embed_dict[c] for c in conds])

    fig, ax = plt.subplots(figsize=(7, 5))
    speed_color = {
        "Speed_Max": "#d62728", "Speed_Mid": "#ff7f0e",
        "Neutral": "#2ca02c", "Accuracy_Mid": "#1f77b4", "Accuracy_Max": "#9467bd"
    }
    for i, (spd, load) in enumerate(conds):
        col    = speed_color.get(spd, "#333")
        marker = "o" if load == "NoLoad" else "^"
        ax.scatter(pts[i, 0], pts[i, 1], c=col, marker=marker, s=100, zorder=3)
        ax.annotate(f"{spd[:6]}\n{load[:2]}",
                    (pts[i, 0], pts[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("SBERT Instruction Embeddings (PCA-2 projection)")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="NoLoad", markerfacecolor="#888", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="Load",   markerfacecolor="#888", markersize=8),
    ] + [Line2D([0], [0], marker="s", color="w", label=k,
                markerfacecolor=v, markersize=8)
         for k, v in speed_color.items()]
    ax.legend(handles=legend_elements, fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tier1_features(out_path):
    """Visualise Tier-1 and Tier-2 feature values per condition."""
    penalties = [5, 10, 20, 40, 80]
    labels    = ["Speed\nMax", "Speed\nMid", "Neutral", "Acc\nMid", "Acc\nMax"]
    reward    = 20

    log_rp     = [np.log(reward / p) for p in penalties]
    tau_acc    = [p / (reward + p) for p in penalties]
    log_pen    = [np.log(p) for p in penalties]
    delta_a    = [np.sqrt((p / reward + 1) / 2) - 1 for p in penalties]
    r_net      = [(reward - p) / (reward + p) for p in penalties]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    def _bar(ax, vals, title, ylabel, hline=None):
        colors = ["#d62728" if v < 0 else "#2ca02c" if v > 0 else "#888" for v in vals]
        ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle="--")
        for i, v in enumerate(vals):
            ax.text(i, v + (0.01 if v >= 0 else -0.04),
                    f"{v:.2f}", ha="center", fontsize=8)

    _bar(axes[0, 0], log_pen, "log(Penalty)  [Tier 0]", "value")
    _bar(axes[0, 1], log_rp,  "log(R/P)  [Tier 1]", "value", hline=0)
    _bar(axes[0, 2], tau_acc, "Break-even Accuracy τ  [Tier 1]", "proportion")

    ord_vals = [0.00, 0.25, 0.50, 0.75, 1.00]
    _bar(axes[1, 0], ord_vals, "Strategy Ordinal  [Tier 1]", "value")
    _bar(axes[1, 1], delta_a,  "Δa* (DDM boundary shift)  [Tier 2]", "value", hline=0)
    _bar(axes[1, 2], r_net,    "r_net (net EV normalised)  [Tier 2]", "value", hline=0)

    fig.suptitle("Task Feature Values by Speed-Accuracy Condition (NoLoad)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_predictions(Y, preds_dict, out_path, target_idx=0,
                             target_name="RT_mean"):
    """Scatter actual vs predicted for each model (best tier per model family)."""
    n = len(preds_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    flat_ax = axes.flatten()
    yt = Y[:, target_idx]

    for ax_i, (label, yp_full) in enumerate(preds_dict.items()):
        ax = flat_ax[ax_i]
        yp = yp_full[:, target_idx]
        mask = np.isfinite(yp) & np.isfinite(yt)
        rho  = spearmanr(yt[mask], yp[mask]).correlation
        r2   = r2_score(yt[mask], yp[mask])
        ax.scatter(yt[mask], yp[mask], alpha=0.25, s=8,
                   c=PALETTE.get(label, "#888"), rasterized=True)
        mn, mx = yt[mask].min(), yt[mask].max()
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
        ax.set_xlabel(f"Actual {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(f"{label}\nρ={rho:.3f}  R²={r2:.3f}")
        ax.grid(True, alpha=0.2)

    for ax_i in range(len(preds_dict), len(flat_ax)):
        flat_ax[ax_i].set_visible(False)

    fig.suptitle(f"Actual vs Predicted — {target_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(df, out_path):
    """Heatmap: models × targets, colour = Spearman ρ."""
    targets = ["RT_mean_rho", "RT_sd_rho", "ACC_rho"]
    pivot   = df.set_index("display_name")[targets].astype(float)
    pivot.columns = ["RT mean", "RT sd", "ACC"]

    fig, ax = plt.subplots(figsize=(6, max(3, len(pivot) * 0.5 + 1)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-0.1, vmax=0.5,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Spearman ρ — All Models × All Targets")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0_global = time.time()

    print("Loading data...")
    merged, feature_cols = load_data()

    X_eeg_raw  = merged[feature_cols].values.astype(np.float64)
    X_eeg      = _impute(X_eeg_raw)
    Y_raw      = merged[["Lexical.RT_mean", "Lexical.RT_sd", "Lexical.ACC"]].values
    Y          = _impute(Y_raw.astype(np.float64))
    groups     = merged["subject"].values
    cond_idx   = merged["cond_idx"].values.astype(int)

    n_subj  = len(np.unique(groups))
    n_eeg   = X_eeg.shape[1]
    print(f"  {len(Y)} rows | {n_subj} subjects | {n_eeg} EEG features")
    print(f"  Device: {DEVICE}")

    # -----------------------------------------------------------------------
    # Build SBERT embeddings (Tier 3) once
    # -----------------------------------------------------------------------
    sbert_embeds = None
    pca_obj      = None
    if not args.skip_sbert:
        print("\nBuilding SBERT embeddings (Tier 3)...")
        try:
            sbert_embeds, pca_obj = build_sbert_embeddings(n_components=8)
            print("  Done.")
        except Exception as e:
            print(f"  WARNING: SBERT failed ({e}). Tier 3 will be skipped.")

    # -----------------------------------------------------------------------
    # Build Tier-1 warm-start weights for Tier-4 learned embedding
    # -----------------------------------------------------------------------
    init_weights = build_tier1_init_weights()  # (10, 5)

    # -----------------------------------------------------------------------
    # Build all task arrays
    # -----------------------------------------------------------------------
    task_arrays = {}
    for t in [0, 1, 2]:
        task_arrays[t] = _impute(build_task_arrays(merged, tier=t))
    if sbert_embeds is not None:
        task_arrays[3] = _impute(build_task_arrays(merged, tier=3,
                                                    sbert_embeds=sbert_embeds))

    # For XGBoost input: concat [X_eeg | X_task]
    all_results  = []
    all_preds    = {}   # label → preds array (for scatter plots)

    epochs = 50 if args.fast else DL_EPOCHS

    # -----------------------------------------------------------------------
    # XGBoost: Tiers 0–3
    # -----------------------------------------------------------------------
    xgb_tiers = [0, 1, 2]
    if sbert_embeds is not None:
        xgb_tiers.append(3)

    for t in xgb_tiers:
        label = f"t{t}_xgb"
        dname = MODEL_DISPLAY[label]
        print(f"\n[XGBoost Tier {t}] {dname}")
        X_in  = np.hstack([X_eeg, task_arrays[t]])
        preds = run_xgboost(X_in, Y, groups)
        m     = metrics(Y, preds)
        all_results.append({"key": label, "display_name": dname, **m})
        all_preds[dname] = preds
        print(f"  RT_mean rho={m['RT_mean_rho']:+.4f}  RT_sd rho={m['RT_sd_rho']:+.4f}  "
              f"ACC rho={m['ACC_rho']:+.4f}")

    # -----------------------------------------------------------------------
    # MLP: Tiers 0–3 (tabular task features concatenated)
    # -----------------------------------------------------------------------
    mlp_tiers = [0, 1, 2]
    if sbert_embeds is not None:
        mlp_tiers.append(3)

    for t in mlp_tiers:
        label = f"t{t}_mlp"
        dname = MODEL_DISPLAY[label]
        print(f"\n[MLP Tier {t}] {dname}")
        preds = run_mlp(X_eeg, task_arrays[t], Y, groups,
                        epochs=epochs, use_learned_emb=False)
        m     = metrics(Y, preds)
        all_results.append({"key": label, "display_name": dname, **m})
        all_preds[dname] = preds
        print(f"  RT_mean rho={m['RT_mean_rho']:+.4f}  RT_sd rho={m['RT_sd_rho']:+.4f}  "
              f"ACC rho={m['ACC_rho']:+.4f}")

    # -----------------------------------------------------------------------
    # MLP Tier 4: learned embedding (warm-started from Tier 1)
    # -----------------------------------------------------------------------
    print(f"\n[MLP Tier 4] learned embedding (warm-start)")
    preds = run_mlp(X_eeg, task_arrays[0], Y, groups,
                    cond_idx=cond_idx, epochs=epochs,
                    use_learned_emb=True, init_weights=init_weights)
    m = metrics(Y, preds)
    all_results.append({"key": "t4_mlp", "display_name": "MLP-T4", **m})
    all_preds["MLP-T4"] = preds
    print(f"  RT_mean rho={m['RT_mean_rho']:+.4f}  RT_sd rho={m['RT_sd_rho']:+.4f}  "
          f"ACC rho={m['ACC_rho']:+.4f}")

    # -----------------------------------------------------------------------
    # Save results CSV
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "task_embedding_results.csv", index=False)
    print(f"\nResults saved → {OUT_DIR / 'task_embedding_results.csv'}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("Generating plots...")

    # 1. Feature value visualisation (no model needed)
    plot_tier1_features(OUT_DIR / "fig1_task_feature_values.png")
    print("  fig1_task_feature_values.png")

    # 2. SBERT PCA scatter
    if sbert_embeds is not None and pca_obj is not None:
        plot_sbert_pca(sbert_embeds, pca_obj, OUT_DIR / "fig2_sbert_pca.png")
        print("  fig2_sbert_pca.png")

    # 3. Spearman ρ bar chart
    plot_rho_comparison(results_df, OUT_DIR / "fig3_rho_comparison.png")
    print("  fig3_rho_comparison.png")

    # 4. Δρ vs Tier-0 baseline
    plot_delta_from_baseline(results_df, OUT_DIR / "fig4_delta_rho.png")
    print("  fig4_delta_rho.png")

    # 5. Heatmap
    plot_heatmap(results_df, OUT_DIR / "fig5_heatmap.png")
    print("  fig5_heatmap.png")

    # 6. Scatter: best XGB vs best MLP for RT_mean
    # Pick best tier per model family
    xgb_best = results_df[results_df["key"].str.endswith("xgb")].nlargest(
        1, "RT_mean_rho")["display_name"].values[0]
    mlp_best = results_df[results_df["key"].str.endswith("mlp")].nlargest(
        1, "RT_mean_rho")["display_name"].values[0]
    scatter_dict = {k: all_preds[k] for k in [xgb_best, mlp_best]
                    if k in all_preds}
    plot_scatter_predictions(Y, scatter_dict,
                              OUT_DIR / "fig6_scatter_rtmean.png",
                              target_idx=0, target_name="RT_mean (ms)")
    print("  fig6_scatter_rtmean.png")

    # 7. Scatter: best XGB vs best MLP for ACC
    scatter_dict_acc = {k: all_preds[k] for k in [xgb_best, mlp_best]
                        if k in all_preds}
    plot_scatter_predictions(Y, scatter_dict_acc,
                              OUT_DIR / "fig7_scatter_acc.png",
                              target_idx=2, target_name="ACC (proportion)")
    print("  fig7_scatter_acc.png")

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    write_report(results_df, sbert_embeds, pca_obj, OUT_DIR / "task_embedding_report.md")
    print(f"\nReport: {OUT_DIR / 'task_embedding_report.md'}")

    elapsed = (time.time() - t0_global) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report(results_df, sbert_embeds, pca_obj, path):
    lines = [
        "# Task Embedding Experiment Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Overview",
        "",
        "Compares five task encoding strategies for EEG + task → RT prediction:",
        "",
        "| Tier | Label | Dim | Description |",
        "|------|-------|-----|-------------|",
        "| 0 | Baseline | 2 | [load, log(penalty)] — current approach |",
        "| 1 | SDT/EV | 5 | [load, log(R/P), τ_acc, strategy_ord, load×log(R/P)] |",
        "| 2 | DDM | 8 | Tier1 + [r_net, Δa*, rt_direction] |",
        "| 3 | SBERT | 8 | Sentence-BERT instruction text → PCA-8 |",
        "| 4 | Learned | 8 | nn.Embedding(10,8) warm-started from Tier1 |",
        "",
        "Models tested: **XGBoost** (Tiers 0–3), **MLP** (Tiers 0–4)",
        "",
        "Cross-validation: 5-fold GroupKFold (subject-level splits)",
        "",
    ]

    # Results table
    lines += [
        "## Results: Spearman ρ (RT mean, RT sd, ACC)",
        "",
        "| Embedding | RT_mean ρ | RT_sd ρ | ACC ρ |",
        "|-----------|-----------|---------|-------|",
    ]
    for _, row in results_df.iterrows():
        r1 = f"{row['RT_mean_rho']:+.4f}" if np.isfinite(row['RT_mean_rho']) else "N/A"
        r2 = f"{row['RT_sd_rho']:+.4f}"   if np.isfinite(row['RT_sd_rho'])   else "N/A"
        r3 = f"{row['ACC_rho']:+.4f}"     if np.isfinite(row['ACC_rho'])     else "N/A"
        lines.append(f"| {row['display_name']} | {r1} | {r2} | {r3} |")

    lines += [""]

    # Delta table
    lines += [
        "## Δρ vs Tier-0 Baseline (RT_mean)",
        "",
        "| Embedding | Δρ (RT_mean) | Δρ (RT_sd) | Δρ (ACC) |",
        "|-----------|-------------|------------|----------|",
    ]
    b0_xgb_rho = results_df.loc[results_df["key"] == "t0_xgb", "RT_mean_rho"].values
    b0_mlp_rho = results_df.loc[results_df["key"] == "t0_mlp", "RT_mean_rho"].values
    b0_xgb_sd  = results_df.loc[results_df["key"] == "t0_xgb", "RT_sd_rho"].values
    b0_mlp_sd  = results_df.loc[results_df["key"] == "t0_mlp", "RT_sd_rho"].values
    b0_xgb_acc = results_df.loc[results_df["key"] == "t0_xgb", "ACC_rho"].values
    b0_mlp_acc = results_df.loc[results_df["key"] == "t0_mlp", "ACC_rho"].values

    for _, row in results_df.iterrows():
        if row["key"] in ("t0_xgb", "t0_mlp"):
            continue
        base_rt  = b0_xgb_rho[0] if row["key"].endswith("xgb") else b0_mlp_rho[0]
        base_sd  = b0_xgb_sd[0]  if row["key"].endswith("xgb") else b0_mlp_sd[0]
        base_acc = b0_xgb_acc[0] if row["key"].endswith("xgb") else b0_mlp_acc[0]
        drt  = row["RT_mean_rho"] - base_rt  if np.isfinite(row["RT_mean_rho"])  else np.nan
        dsd  = row["RT_sd_rho"]   - base_sd  if np.isfinite(row["RT_sd_rho"])    else np.nan
        dacc = row["ACC_rho"]     - base_acc if np.isfinite(row["ACC_rho"])       else np.nan
        s_rt  = f"{drt:+.4f}" if np.isfinite(drt)  else "N/A"
        s_sd  = f"{dsd:+.4f}" if np.isfinite(dsd)  else "N/A"
        s_acc = f"{dacc:+.4f}" if np.isfinite(dacc) else "N/A"
        lines.append(f"| {row['display_name']} | {s_rt} | {s_sd} | {s_acc} |")

    # SBERT variance explained
    if pca_obj is not None:
        lines += [
            "",
            "## SBERT PCA: Explained Variance",
            "",
            "| PC | Explained Variance | Cumulative |",
            "|----|-------------------|------------|",
        ]
        cum = 0.0
        for i, v in enumerate(pca_obj.explained_variance_ratio_):
            cum += v
            lines.append(f"| PC{i+1} | {v*100:.1f}% | {cum*100:.1f}% |")

    lines += [
        "",
        "## Figures",
        "",
        "| Figure | Description |",
        "|--------|-------------|",
        "| fig1_task_feature_values.png | Tier-1 and Tier-2 feature values by condition |",
        "| fig2_sbert_pca.png | SBERT instruction embeddings (PCA-2) |",
        "| fig3_rho_comparison.png | Spearman ρ bar chart for all tiers/models |",
        "| fig4_delta_rho.png | Δρ gain over Tier-0 baseline |",
        "| fig5_heatmap.png | Heatmap: models × targets |",
        "| fig6_scatter_rtmean.png | Actual vs predicted RT_mean (best tier per model) |",
        "| fig7_scatter_acc.png | Actual vs predicted ACC (best tier per model) |",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Embedding Experiment")
    parser.add_argument("--skip-sbert", action="store_true",
                        help="Skip Tier 3 (SBERT) if not needed")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce epochs to 50 for quick testing")
    args = parser.parse_args()
    main(args)
