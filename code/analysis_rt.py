#!/usr/bin/env python3
"""
EEG -> Decision Behavior: Resting EEG vs EEG + Task
====================================================
Compares resting-state EEG alone vs EEG + task condition for
predicting RT and accuracy at subject x condition level.

Models:
  Level 1: Ridge, XGBoost          (classical ML)
  Level 2: MLP                     (deep learning baseline)
  Level 3: Neural SDE              (learned evidence accumulation)
  Level 4: ROI-GNN + Neural SDE    (topology-preserving encoder)

For each model, two input modes:
  A) EEG only    -> RT_mean, RT_sd, ACC
  B) EEG + Task  -> RT_mean, RT_sd, ACC
  Delta(B - A) = added value of task information

Task encoding: structured — Load (0/1) + log(Penalty)

Usage:
    python code/analysis_rt.py
    python code/analysis_rt.py --gnn
    python code/analysis_rt.py --gnn --sde-epochs 200
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
_CANDIDATES = [
    Path("."),
    Path("/storage/project/r-nimam6-0/hkim3239/cogegg"),
    Path("/Users/hkim3239/GaTech Dropbox/Hyunju Kim/EEG"),
]


def _find_base():
    env = os.environ.get("EEG_BASE_DIR")
    if env:
        return Path(env)
    for d in _CANDIDATES:
        if (d / "data" / "resting_subject_features.csv").exists():
            return d
    return Path(".")


BASE_DIR = _find_base()
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "rt_prediction"

N_FOLDS = 5
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep learning
DL_EPOCHS = 100
DL_LR = 1e-3
DL_PATIENCE = 15
DL_BATCH = 128

# SDE simulation
SDE_N_SIM = 16
SDE_DT = 0.02
SDE_MAX_T = 1.0
SDE_TEMP = 20.0
SDE_DETACH_K = 10

# ROI graph
ROI_NAMES = [
    "prefrontal", "frontal", "central", "posterior",
    "occipital", "left_temporal", "right_temporal",
]
ROI_EDGES = [
    ("prefrontal", "frontal"),
    ("frontal", "central"),
    ("frontal", "left_temporal"),
    ("frontal", "right_temporal"),
    ("central", "posterior"),
    ("central", "left_temporal"),
    ("central", "right_temporal"),
    ("posterior", "occipital"),
    ("posterior", "left_temporal"),
    ("posterior", "right_temporal"),
    ("occipital", "left_temporal"),
    ("occipital", "right_temporal"),
    ("left_temporal", "right_temporal"),
]
NODE_SUFFIXES = [
    "aperiodic_exponent", "aperiodic_offset",
    "alpha_peak_cf", "alpha_peak_pw", "alpha_peak_bw",
    "beta_peak_cf", "beta_peak_pw", "beta_peak_bw",
    "delta_peak_cf", "delta_peak_pw", "delta_peak_bw",
    "theta_peak_cf", "theta_peak_pw", "theta_peak_bw",
    "perm_entropy", "sample_entropy",
    "alpha", "beta", "theta", "theta_beta_ratio",
]

TARGET_NAMES = ["RT_mean", "RT_sd", "ACC"]


# =============================================================================
# Data
# =============================================================================
def _impute_nan(X):
    """Column-wise median imputation (in-place)."""
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any():
            med = np.nanmedian(X[:, j])
            X[m, j] = med if np.isfinite(med) else 0
    return X


def load_data():
    """Load and merge EEG + DDM at subject x condition level."""
    # EEG
    eeg = pd.read_csv(DATA_DIR / "resting_subject_features.csv")
    if "Subject" in eeg.columns:
        eeg = eeg.rename(columns={"Subject": "subject"})
    drop = ["processing_failed", "error", "qc_flag", "n_bad_channels",
            "n_epochs_total", "n_epochs_kept", "flag_low_epochs"]
    eeg = eeg.drop(columns=[c for c in drop if c in eeg.columns], errors="ignore")
    eeg = eeg[eeg["subject"].notna()]

    # DDM + behavioral summary
    ddm = pd.read_csv(DATA_DIR / "DDM_Scores.csv")
    keep = ["Subject", "Detailed_Condition", "Penalty", "Load_Condition",
            "Lexical.ACC", "Lexical.RT_mean", "Lexical.RT_sd", "a", "v", "t0"]
    ddm = ddm[[c for c in keep if c in ddm.columns]].rename(
        columns={"Subject": "subject", "Detailed_Condition": "condition"})

    merged = ddm.merge(eeg, on="subject", how="inner")
    merged = merged.dropna(subset=["Lexical.RT_mean"])

    feature_cols = [c for c in eeg.columns if c != "subject"]

    # Task encoding: structured
    merged["task_load"] = (merged["Load_Condition"] == "Load").astype(float)
    merged["task_log_penalty"] = np.log(merged["Penalty"].astype(float))

    return merged, feature_cols


def prepare_arrays(merged, feature_cols):
    """Prepare numpy arrays for modeling."""
    X_eeg = _impute_nan(merged[feature_cols].values.astype(np.float64))
    X_task = merged[["task_load", "task_log_penalty"]].values.astype(np.float64)
    Y = _impute_nan(
        merged[["Lexical.RT_mean", "Lexical.RT_sd", "Lexical.ACC"]]
        .values.astype(np.float64)
    )
    groups = merged["subject"].values
    return X_eeg, X_task, Y, groups


def prepare_roi_graph(merged, feature_cols):
    """Build ROI node features, global features, and adjacency matrix."""
    N = len(merged)
    n_nodes = len(ROI_NAMES)
    n_nf = len(NODE_SUFFIXES)

    # Node features: (N, 7, 20)
    node_feat = np.zeros((N, n_nodes, n_nf), dtype=np.float64)
    for ni, roi in enumerate(ROI_NAMES):
        for fi, suf in enumerate(NODE_SUFFIXES):
            col = f"{roi}_{suf}"
            if col in merged.columns:
                node_feat[:, ni, fi] = merged[col].values
    # Impute per-node per-feature
    for ni in range(n_nodes):
        for fi in range(n_nf):
            v = node_feat[:, ni, fi]
            m = np.isnan(v)
            if m.any():
                med = np.nanmedian(v)
                node_feat[m, ni, fi] = med if np.isfinite(med) else 0

    # Global features (everything not assigned to a ROI node)
    roi_assigned = {f"{r}_{s}" for r in ROI_NAMES for s in NODE_SUFFIXES}
    global_cols = [c for c in feature_cols if c not in roi_assigned]
    global_feat = _impute_nan(merged[global_cols].values.astype(np.float64))

    # Connectivity-weighted adjacency
    roi2idx = {r: i for i, r in enumerate(ROI_NAMES)}
    adj = np.eye(n_nodes, dtype=np.float32)
    for r1, r2 in ROI_EDGES:
        i, j = roi2idx[r1], roi2idx[r2]
        cvals = []
        for band in ["alpha", "beta", "delta", "theta"]:
            for pat in [f"conn_{r1}_{r2}_{band}", f"conn_{r2}_{r1}_{band}"]:
                if pat in merged.columns:
                    v = np.nanmean(merged[pat].values)
                    if np.isfinite(v):
                        cvals.append(abs(v))
        w = np.mean(cvals) if cvals else 1.0
        adj[i, j] = w
        adj[j, i] = w

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    D_inv = np.diag(deg ** -0.5)
    adj_norm = torch.FloatTensor(D_inv @ adj @ D_inv)

    return node_feat, global_feat, adj_norm


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true, y_pred):
    out = {}
    for i, name in enumerate(TARGET_NAMES):
        yt, yp = y_true[:, i], y_pred[:, i]
        mask = np.isfinite(yp) & np.isfinite(yt)
        if mask.sum() < 10:
            out[f"{name}_r2"] = np.nan
            out[f"{name}_mae"] = np.nan
            out[f"{name}_rho"] = np.nan
        else:
            out[f"{name}_r2"] = r2_score(yt[mask], yp[mask])
            out[f"{name}_mae"] = mean_absolute_error(yt[mask], yp[mask])
            out[f"{name}_rho"] = spearmanr(yt[mask], yp[mask]).correlation
    return out


# =============================================================================
# Level 1: Classical ML
# =============================================================================
def run_classical(X, Y, groups, model_type="ridge"):
    gkf = GroupKFold(n_splits=N_FOLDS)
    preds = np.full_like(Y, np.nan)
    for tr, te in gkf.split(X, groups=groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for oi in range(3):
            if model_type == "ridge":
                m = RidgeCV(alphas=np.logspace(-3, 3, 50))
            else:
                m = XGBRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=RANDOM_STATE, verbosity=0)
            m.fit(Xtr, Y[tr, oi])
            preds[te, oi] = m.predict(Xte)
    return preds


# =============================================================================
# Level 2: MLP
# =============================================================================
class MLPModel(nn.Module):
    def __init__(self, n_in, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Level 3 & 4: SDE base
# =============================================================================
class SDEBase(nn.Module):
    """Shared SDE simulation logic for NeuralSDE and GNN+SDE."""

    def _init_sde(self, latent):
        self._sde_max_steps = int(SDE_MAX_T / SDE_DT)
        # Combined drift+diffusion network (one forward pass instead of two)
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent + 2, 64), nn.Tanh(), nn.Linear(64, 2))
        self.boundary_head = nn.Linear(latent, 1)
        self.ndt_head = nn.Linear(latent, 1)
        # Learnable output affine (adapts SDE's raw ms/acc scale to normalized Y)
        self.out_scale = nn.Parameter(torch.tensor([0.001, 0.001, 0.1]))
        self.out_bias = nn.Parameter(torch.zeros(3))
        # Stable initialization: drift~0.5, diff~0 (softplus(0)+0.1=0.79)
        nn.init.constant_(self.dynamics_net[-1].bias[:1], 0.5)
        nn.init.zeros_(self.dynamics_net[-1].bias[1:])

    def _sde_forward(self, z):
        """Run soft-boundary SDE from latent z. Returns (B, 3)."""
        boundary = F.softplus(self.boundary_head(z)).squeeze(-1) + 0.3
        ndt = F.softplus(self.ndt_head(z)).squeeze(-1) + 0.05
        half_b = boundary / 2

        B = z.shape[0]
        S = SDE_N_SIM
        sqrt_dt = SDE_DT ** 0.5

        x = torch.zeros(B, S, device=z.device)
        log_surv = torch.zeros(B, S, device=z.device)
        exp_rt = torch.zeros(B, S, device=z.device)
        exp_corr = torch.zeros(B, S, device=z.device)
        z_exp = z.unsqueeze(1).expand(-1, S, -1)  # (B, S, latent)

        for step in range(self._sde_max_steps):
            # Truncate gradient chain for memory
            if step > 0 and step % SDE_DETACH_K == 0:
                x = x.detach()

            t_val = step * SDE_DT
            inp = torch.cat([
                x.unsqueeze(-1),
                torch.full((B, S, 1), t_val, device=z.device),
                z_exp,
            ], dim=-1)  # (B, S, latent+2)

            dyn = self.dynamics_net(inp)                                   # (B, S, 2)
            drift = dyn[..., 0].clamp(-5, 5)                              # (B, S)
            diff = F.softplus(dyn[..., 1]) + 0.1                          # (B, S)

            noise = torch.randn(B, S, device=z.device)
            x = (x + drift * SDE_DT + diff * sqrt_dt * noise).clamp(-10, 10)

            # Soft boundary: hazard = P(cross | at this state)
            dist = x.abs() - half_b.unsqueeze(1)
            hazard = torch.sigmoid(SDE_TEMP * dist).clamp(0, 0.99)

            surv_before = torch.exp(log_surv.clamp(min=-50))
            cross_p = surv_before * hazard  # P(first crossing at this step)
            log_surv = log_surv + torch.log1p(-hazard)

            t_now = (step + 1) * SDE_DT
            exp_rt = exp_rt + cross_p * t_now
            exp_corr = exp_corr + cross_p * (x > 0).float()

        # Remaining survival -> max RT
        remaining = torch.exp(log_surv.clamp(min=-50))
        exp_rt = exp_rt + remaining * self._sde_max_steps * SDE_DT
        exp_corr = exp_corr + remaining * 0.5

        # Convert to ms, add non-decision time
        rt_ms = (exp_rt + ndt.unsqueeze(1)) * 1000  # (B, S)

        raw = torch.stack([
            rt_ms.mean(dim=1),       # RT_mean
            rt_ms.std(dim=1) + 1e-3, # RT_sd (prevent zero)
            exp_corr.mean(dim=1),    # ACC
        ], dim=1)  # (B, 3)

        out = raw * self.out_scale + self.out_bias
        return torch.nan_to_num(out, nan=0.0)


# =============================================================================
# Level 3: Neural SDE
# =============================================================================
class NeuralSDE(SDEBase):
    """
    Evidence accumulation with learned dynamics.

    DDM:        dx = v * dt + 1 * dW         (v constant, noise = 1)
    Neural SDE: dx = f(x,t,z)*dt + g(x,t,z)*dW  (f,g are neural nets)

    z = encoder(EEG [+ task]) conditions the drift and diffusion.
    Soft boundary enables end-to-end gradient flow.
    """

    def __init__(self, input_dim, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, latent), nn.LayerNorm(latent), nn.GELU(),
        )
        self._init_sde(latent)

    def forward(self, x):
        z = self.encoder(x)
        return self._sde_forward(z)


# =============================================================================
# Level 4: ROI-GNN + Neural SDE
# =============================================================================
class GNNwithSDE(SDEBase):
    """
    ROI graph encoder -> Neural SDE decoder.

    7 brain ROIs as graph nodes, connectivity-weighted adjacency.
    GCN propagation preserves spatial topology.
    Input is flattened for compatibility with generic training loop.
    """

    def __init__(self, flat_dim, n_node_feat, n_nodes, n_global, n_task, adj,
                 latent=64):
        super().__init__()
        self.n_node_feat = n_node_feat
        self.n_nodes = n_nodes
        self.n_global = n_global
        self.n_task = n_task
        self.register_buffer("adj", adj)

        # 2-layer GCN
        self.W1 = nn.Linear(n_node_feat, 64)
        self.ln1 = nn.LayerNorm(64)
        self.W2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)

        # Fuse graph embedding + global features + task
        self.fuse = nn.Sequential(
            nn.Linear(64 + n_global + n_task, latent),
            nn.LayerNorm(latent), nn.GELU(),
        )
        self._init_sde(latent)

    def forward(self, x_flat):
        B = x_flat.shape[0]
        # Unpack flat input -> (node_feat, global_feat, task_feat)
        o = 0
        nf = x_flat[:, o:o + self.n_nodes * self.n_node_feat]
        nf = nf.reshape(B, self.n_nodes, self.n_node_feat)
        o += self.n_nodes * self.n_node_feat

        gf = x_flat[:, o:o + self.n_global]
        o += self.n_global

        tf = x_flat[:, o:o + self.n_task]

        # GCN: H = GELU(LN(A @ X @ W))
        A = self.adj.unsqueeze(0).expand(B, -1, -1)
        h = F.gelu(self.ln1(self.W1(torch.bmm(A, nf))))
        h = F.gelu(self.ln2(self.W2(torch.bmm(A, h))))
        g = h.mean(dim=1)  # global mean pool -> (B, 64)

        z = self.fuse(torch.cat([g, gf, tf], dim=-1))
        return self._sde_forward(z)


# =============================================================================
# Training
# =============================================================================
def _train_model(model, X_t, Y_t, X_v, Y_v,
                 epochs=DL_EPOCHS, lr=DL_LR, patience=DL_PATIENCE,
                 batch_size=DL_BATCH):
    """Train with mini-batch, early stopping, grad clipping."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5)
    best_loss, wait, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            batch = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X_t[batch])
            loss = F.mse_loss(pred, Y_t[batch])
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = F.mse_loss(val_pred, Y_v).item()
        if np.isnan(val_loss):
            val_loss = float("inf")
        sched.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_neural(X, Y, groups, model_fn, epochs=DL_EPOCHS,
               n_eval_avg=1, label=""):
    """Generic CV runner for any neural model (MLP / SDE / GNN+SDE)."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    preds = np.full_like(Y, np.nan)

    for fold, (tr, te) in enumerate(gkf.split(X, groups=groups)):
        t0 = time.time()

        # Scale X
        sc = StandardScaler().fit(X[tr])
        Xtr_sc = sc.transform(X[tr])
        Xte_sc = sc.transform(X[te])

        # Normalize Y (per-fold)
        y_mu = Y[tr].mean(axis=0)
        y_sig = Y[tr].std(axis=0)
        y_sig[y_sig < 1e-8] = 1.0
        Ytr_n = (Y[tr] - y_mu) / y_sig

        # Train / val split
        n_val = max(int(len(tr) * 0.15), 1)
        perm = np.random.RandomState(RANDOM_STATE + fold).permutation(len(tr))
        ti, vi = perm[n_val:], perm[:n_val]

        X_t = torch.FloatTensor(Xtr_sc[ti]).to(DEVICE)
        Y_t = torch.FloatTensor(Ytr_n[ti]).to(DEVICE)
        X_v = torch.FloatTensor(Xtr_sc[vi]).to(DEVICE)
        Y_v = torch.FloatTensor(Ytr_n[vi]).to(DEVICE)

        model = model_fn(Xtr_sc.shape[1]).to(DEVICE)
        model = _train_model(model, X_t, Y_t, X_v, Y_v, epochs=epochs)

        # Predict (average multiple passes for stochastic SDE)
        X_test = torch.FloatTensor(Xte_sc).to(DEVICE)
        with torch.no_grad():
            runs = [model(X_test).cpu().numpy() for _ in range(n_eval_avg)]
        pred_n = np.mean(runs, axis=0)
        preds[te] = pred_n * y_sig + y_mu

        dt = time.time() - t0
        print(f"    Fold {fold + 1}/{N_FOLDS}: {dt:.1f}s")

    return preds


# =============================================================================
# Report
# =============================================================================
def write_report(results_df, path):
    lines = [
        "# EEG to Decision Behavior: Analysis Report",
        "",
        "## Experimental Design",
        "",
        "This analysis tests whether resting-state EEG features can predict",
        "decision-making behavior (RT and accuracy), and whether adding task",
        "condition information improves prediction.",
        "",
        "### Data",
        "",
        "- **EEG**: Resting-state features (~234 features per subject)",
        "  - Aperiodic (1/f exponent, offset) per ROI",
        "  - Periodic (FOOOF peak CF, PW, BW for delta/theta/alpha/beta) per ROI",
        "  - Entropy (permutation, sample) per ROI",
        "  - Connectivity (imaginary coherence between ROI pairs)",
        "  - Global (IAF, band powers, asymmetry, frontal-parietal coherence)",
        "- **Task conditions**: 2 x 5 factorial (Load x Speed-Accuracy emphasis)",
        "  - Load: Load vs NoLoad (working memory manipulation)",
        "  - Speed-Accuracy: Penalty = {5, 10, 20, 40, 80}",
        "  - Encoded as: Load (0/1) + log(Penalty)",
        "- **Targets**: RT_mean (ms), RT_sd (ms), ACC (proportion)",
        "  - Per subject x condition (~10 conditions per subject)",
        "",
        "### Comparisons",
        "",
        "| Input Mode | Features | Question |",
        "|-----------|----------|----------|",
        "| EEG only | Resting EEG (234d) | Can brain state predict behavior? |",
        "| EEG + Task | EEG + Load(0/1) + log(Penalty) | Does task context help? |",
        "",
        f"**CV**: {N_FOLDS}-fold GroupKFold (subject-level split, no data leakage)",
        "",
        "## Models",
        "",
        "| Level | Model | Architecture |",
        "|-------|-------|-------------|",
        "| 1 | Ridge | L2-regularized linear regression |",
        "| 1 | XGBoost | Gradient-boosted trees (depth=3, 100 trees) |",
        "| 2 | MLP | 128->64->3 with LayerNorm, GELU, Dropout |",
        "| 3 | Neural SDE | Learned drift f(x,t,z) + diffusion g(x,t,z), soft boundary |",
        "| 4 | GNN+SDE | 7-ROI GCN (connectivity-weighted) -> Neural SDE |",
        "",
        "### Neural SDE: Brain-Inspired Evidence Accumulation",
        "",
        "Classical DDM uses fixed parameters:",
        "```",
        "dx = v * dt + 1 * dW     (constant drift v, noise = 1)",
        "```",
        "",
        "Neural SDE learns time- and state-dependent dynamics:",
        "```",
        "dx = f_theta(x, t, z) * dt + g_theta(x, t, z) * dW",
        "```",
        "where z = encoder(EEG [+ task]) is a latent brain-state vector.",
        "",
        "This allows modeling phenomena DDM cannot capture:",
        "- Urgency signals (drift increases with time)",
        "- Attention fluctuations (state-dependent noise)",
        "- Nonlinear evidence accumulation",
        "",
        "### ROI-GNN: Topology-Preserving Encoder",
        "",
        "7 brain ROIs (prefrontal, frontal, central, posterior, occipital,",
        "left/right temporal) form a graph with connectivity-weighted edges.",
        "2-layer GCN propagates information along anatomical adjacency,",
        "preserving spatial relationships that a flat MLP would ignore.",
        "",
        "## Results: RT_mean Prediction",
        "",
        "| Model | Input | R-squared | MAE (ms) | Spearman rho |",
        "|-------|-------|-----------|---------|-------------|",
    ]

    for _, row in results_df.sort_values(["input", "model"]).iterrows():
        r2 = row.get("RT_mean_r2", np.nan)
        mae = row.get("RT_mean_mae", np.nan)
        rho = row.get("RT_mean_rho", np.nan)
        r2_s = f"{r2:+.4f}" if np.isfinite(r2) else "N/A"
        mae_s = f"{mae:.1f}" if np.isfinite(mae) else "N/A"
        rho_s = f"{rho:+.4f}" if np.isfinite(rho) else "N/A"
        lines.append(f"| {row['model']} | {row['input']} | {r2_s} | {mae_s} | {rho_s} |")

    lines += [
        "",
        "## Results: ACC Prediction",
        "",
        "| Model | Input | R-squared | MAE | Spearman rho |",
        "|-------|-------|-----------|-----|-------------|",
    ]

    for _, row in results_df.sort_values(["input", "model"]).iterrows():
        r2 = row.get("ACC_r2", np.nan)
        mae = row.get("ACC_mae", np.nan)
        rho = row.get("ACC_rho", np.nan)
        r2_s = f"{r2:+.4f}" if np.isfinite(r2) else "N/A"
        mae_s = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        rho_s = f"{rho:+.4f}" if np.isfinite(rho) else "N/A"
        lines.append(f"| {row['model']} | {row['input']} | {r2_s} | {mae_s} | {rho_s} |")

    # Interpretation
    lines += ["", "## Key Findings", ""]

    eeg_only = results_df[results_df["input"] == "eeg_only"]
    eeg_task = results_df[results_df["input"] == "eeg_task"]

    if len(eeg_only) > 0 and len(eeg_task) > 0:
        best_eeg = eeg_only["RT_mean_r2"].max()
        best_task = eeg_task["RT_mean_r2"].max()
        gain = best_task - best_eeg

        best_eeg_model = eeg_only.loc[eeg_only["RT_mean_r2"].idxmax(), "model"]
        best_task_model = eeg_task.loc[eeg_task["RT_mean_r2"].idxmax(), "model"]

        lines.append(f"- **Best EEG-only**: R2={best_eeg:+.4f} ({best_eeg_model})")
        lines.append(f"- **Best EEG+Task**: R2={best_task:+.4f} ({best_task_model})")
        lines.append(f"- **Task information gain**: Delta R2 = {gain:+.4f}")
        lines.append("")

        if gain > 0.05:
            lines.append(
                "Task condition substantially improves prediction. "
                "The model captures condition-specific individual differences "
                "beyond what resting EEG alone provides.")
        elif gain > 0.01:
            lines.append(
                "Task condition provides moderate improvement. "
                "EEG captures between-subject differences, while task info "
                "adds condition-level specificity.")
        else:
            lines.append(
                "Task condition adds minimal improvement. "
                "Prediction is primarily driven by between-subject EEG differences, "
                "not condition-specific modulation.")

    lines += [
        "",
        "## Model Comparison Notes",
        "",
        "- **Classical vs Deep Learning**: If Ridge/XGBoost match or exceed MLP/SDE,",
        "  the EEG-behavior relationship may be approximately linear.",
        "- **Neural SDE vs MLP**: If SDE outperforms MLP, the evidence-accumulation",
        "  inductive bias captures meaningful decision dynamics.",
        "- **GNN+SDE vs flat SDE**: If GNN improves results, spatial brain topology",
        "  provides useful structure beyond treating all features as exchangeable.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================
def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0_global = time.time()

    # Load data
    print("Loading data...")
    merged, feature_cols = load_data()
    X_eeg, X_task, Y, groups = prepare_arrays(merged, feature_cols)
    n_subj = len(np.unique(groups))
    print(f"  Samples: {len(Y)} ({n_subj} subjects x "
          f"{len(Y) // max(n_subj, 1)} conditions)")
    print(f"  EEG features: {X_eeg.shape[1]}, Task features: {X_task.shape[1]}")
    print(f"  Device: {DEVICE}")

    # ROI graph (for GNN)
    node_feat, global_feat, adj = None, None, None
    if args.gnn:
        node_feat, global_feat, adj = prepare_roi_graph(merged, feature_cols)
        print(f"  ROI graph: {len(ROI_NAMES)} nodes, "
              f"{len(ROI_EDGES)} edges, "
              f"node_feat={node_feat.shape[2]}, "
              f"global_feat={global_feat.shape[1]}")

    all_results = []

    for input_mode in ["eeg_only", "eeg_task"]:
        print(f"\n{'=' * 60}")
        print(f"  Input: {input_mode}")
        print(f"{'=' * 60}")

        if input_mode == "eeg_only":
            X_flat = X_eeg.copy()
            X_task_cur = np.zeros_like(X_task)
        else:
            X_flat = np.hstack([X_eeg, X_task])
            X_task_cur = X_task.copy()

        # --- Level 1: Ridge ---
        print("\n  [Level 1] Ridge")
        preds = run_classical(X_flat, Y, groups, "ridge")
        m = compute_metrics(Y, preds)
        all_results.append({"model": "Ridge", "input": input_mode, **m})
        print(f"    RT_mean  R2={m['RT_mean_r2']:+.4f}  rho={m['RT_mean_rho']:+.4f}  "
              f"MAE={m['RT_mean_mae']:.1f}ms")

        # --- Level 1: XGBoost ---
        print("  [Level 1] XGBoost")
        preds = run_classical(X_flat, Y, groups, "xgboost")
        m = compute_metrics(Y, preds)
        all_results.append({"model": "XGBoost", "input": input_mode, **m})
        print(f"    RT_mean  R2={m['RT_mean_r2']:+.4f}  rho={m['RT_mean_rho']:+.4f}  "
              f"MAE={m['RT_mean_mae']:.1f}ms")

        # --- Level 2: MLP ---
        print("  [Level 2] MLP")
        preds = run_neural(
            X_flat, Y, groups,
            model_fn=lambda n: MLPModel(n, 3),
            epochs=args.dl_epochs, n_eval_avg=1, label="MLP")
        m = compute_metrics(Y, preds)
        all_results.append({"model": "MLP", "input": input_mode, **m})
        print(f"    RT_mean  R2={m['RT_mean_r2']:+.4f}  rho={m['RT_mean_rho']:+.4f}  "
              f"MAE={m['RT_mean_mae']:.1f}ms")

        # --- Level 3: Neural SDE ---
        print("  [Level 3] Neural SDE")
        preds = run_neural(
            X_flat, Y, groups,
            model_fn=lambda n: NeuralSDE(n),
            epochs=args.sde_epochs, n_eval_avg=3, label="NeuralSDE")
        m = compute_metrics(Y, preds)
        all_results.append({"model": "NeuralSDE", "input": input_mode, **m})
        print(f"    RT_mean  R2={m['RT_mean_r2']:+.4f}  rho={m['RT_mean_rho']:+.4f}  "
              f"MAE={m['RT_mean_mae']:.1f}ms")

        # --- Level 4: GNN + SDE ---
        if args.gnn:
            print("  [Level 4] GNN + SDE")
            X_gnn = np.hstack([
                node_feat.reshape(len(node_feat), -1),
                global_feat,
                X_task_cur,
            ])
            n_nf = len(NODE_SUFFIXES)
            n_nodes = len(ROI_NAMES)
            n_global = global_feat.shape[1]
            n_task = X_task_cur.shape[1]

            def make_gnn(flat_dim):
                return GNNwithSDE(
                    flat_dim, n_nf, n_nodes, n_global, n_task, adj)

            preds = run_neural(
                X_gnn, Y, groups,
                model_fn=make_gnn,
                epochs=args.sde_epochs, n_eval_avg=3, label="GNN+SDE")
            m = compute_metrics(Y, preds)
            all_results.append({"model": "GNN+SDE", "input": input_mode, **m})
            print(f"    RT_mean  R2={m['RT_mean_r2']:+.4f}  rho={m['RT_mean_rho']:+.4f}  "
                  f"MAE={m['RT_mean_mae']:.1f}ms")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "rt_results.csv", index=False)
    write_report(results_df, OUT_DIR / "rt_report.md")

    elapsed = (time.time() - t0_global) / 60
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f} min")
    print(f"Results: {OUT_DIR}")
    print(f"Report:  {OUT_DIR / 'rt_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG -> Decision Behavior Prediction")
    parser.add_argument("--gnn", action="store_true",
                        help="Include ROI-GNN + SDE model (Level 4)")
    parser.add_argument("--sde-epochs", type=int, default=DL_EPOCHS,
                        help="Training epochs for SDE models")
    parser.add_argument("--dl-epochs", type=int, default=DL_EPOCHS,
                        help="Training epochs for MLP")
    args = parser.parse_args()
    main(args)
