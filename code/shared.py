"""
Shared utilities for all analyses.
Defines the 39 ROI-level EEG features and data loading functions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/hkim3239/GaTech Dropbox/Hyunju Kim/EEG")
DATA_DIR = BASE_DIR / "data"

ROIS = ['frontal', 'posterior', 'central', 'left_temporal',
        'right_temporal', 'occipital', 'prefrontal']

FEATURE_DOMAINS = {
    'aperiodic_exponent': 'Aperiodic exponent (1/f slope) per ROI',
    'aperiodic_offset': 'Aperiodic offset per ROI',
    'sample_entropy': 'Sample entropy per ROI',
    'perm_entropy': 'Permutation entropy per ROI',
}


def load_eeg_raw():
    """Load raw EEG features, drop metadata columns."""
    df = pd.read_csv(DATA_DIR / "resting_subject_features.csv")
    if "Subject" in df.columns:
        df = df.rename(columns={"Subject": "subject"})
    meta = ["processing_failed", "error", "qc_flag", "n_bad_channels",
            "n_epochs_total", "n_epochs_kept", "flag_low_epochs"]
    df = df.drop(columns=[c for c in meta if c in df.columns], errors="ignore")
    return df[df["subject"].notna()]


def build_39_features(eeg_raw):
    """Build 39 interpretable ROI-level features from raw EEG."""
    feats = {}

    # ROI × domain (7 × 4 = 28)
    for roi in ROIS:
        for domain in FEATURE_DOMAINS:
            col = f'{roi}_{domain}'
            if col in eeg_raw.columns:
                feats[col] = eeg_raw[col]

    # Global band power (4)
    for bp in ['global_delta', 'global_theta', 'global_alpha', 'global_beta']:
        if bp in eeg_raw.columns:
            feats[bp] = eeg_raw[bp]

    # Scalar features (3)
    for c in ['IAF', 'frontal_theta_beta_ratio', 'temporal_alpha_asymmetry']:
        if c in eeg_raw.columns:
            feats[c] = eeg_raw[c]

    # Mean connectivity per band (4)
    conn_cols = [c for c in eeg_raw.columns if c.startswith('conn_')]
    for band in ['delta', 'theta', 'alpha', 'beta']:
        bc = [c for c in conn_cols if c.endswith(f'_{band}')]
        if bc:
            feats[f'connectivity_{band}'] = eeg_raw[bc].mean(axis=1)

    feat_df = pd.DataFrame(feats)
    feat_df['subject'] = eeg_raw['subject'].values
    return feat_df


def load_ddm_grandmean():
    """Load DDM grand-mean (a, v, t0) per subject."""
    ddm = pd.read_csv(DATA_DIR / "DDM_Scores.csv")
    grand = ddm.groupby("Subject")[["a", "v", "t0"]].mean().reset_index()
    return grand.rename(columns={"Subject": "subject"})


def load_composites():
    """Load cognitive composite scores (WMC, gF, AC, SuS_AC)."""
    df = pd.read_csv(DATA_DIR / "all_Scores_filtered_composites.csv")
    return df.rename(columns={"Subject": "subject"})


def load_individual_scores():
    """Load individual cognitive task scores."""
    df = pd.read_csv(DATA_DIR / "all_Scores_filtered.csv")
    return df.rename(columns={"Subject": "subject"})


def impute_median(X):
    """Impute NaN with column median."""
    X = X.copy()
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_med[j] if np.isfinite(col_med[j]) else 0
    return X


def get_feature_cols(feat_df):
    """Return feature column names (everything except 'subject')."""
    return [c for c in feat_df.columns if c != 'subject']
