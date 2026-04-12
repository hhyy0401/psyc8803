#!/usr/bin/env python3
"""
Analysis 2: Multivariate EEG ↔ DDM Association (dCor + CCA)
============================================================
- Distance Correlation: tests multivariate association (nonlinear OK)
- CCA: extracts shared latent dimensions with interpretable loadings

Output:
    output/analysis2_multivariate/
        dcor_results.csv
        cca_loadings.csv
        multivariate_report.md
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

from shared import (BASE_DIR, load_eeg_raw, build_39_features,
                    load_ddm_grandmean, get_feature_cols, impute_median)

OUT_DIR = BASE_DIR / "output" / "analysis2_multivariate"
N_PERM_DCOR = 500
N_PERM_CCA = 5000


def dcov(X, Y):
    """Distance covariance between matrices X and Y."""
    a = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    b = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    return (A * B).mean()


def dcor(X, Y):
    """Distance correlation."""
    xy = dcov(X, Y)
    xx = dcov(X, X)
    yy = dcov(Y, Y)
    if xx * yy <= 0:
        return 0.0
    return np.sqrt(xy / np.sqrt(xx * yy))


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eeg = build_39_features(load_eeg_raw())
    ddm = load_ddm_grandmean()
    feat_cols = get_feature_cols(eeg)
    merged = eeg.merge(ddm, on='subject', how='inner').dropna(subset=['a', 'v', 't0'])
    N = len(merged)

    X_raw = impute_median(merged[feat_cols].values.astype(np.float64))
    Y_raw = merged[['a', 'v', 't0']].values.astype(np.float64)
    X = StandardScaler().fit_transform(X_raw)
    Y = StandardScaler().fit_transform(Y_raw)

    print(f"N={N}, EEG features={len(feat_cols)}")

    # ===== Distance Correlation =====
    print(f"\n--- Distance Correlation ({N_PERM_DCOR} permutations) ---")
    real_dcor = dcor(X, Y)
    print(f"  dCor = {real_dcor:.4f}")

    perm_dcors = []
    for seed in range(N_PERM_DCOR):
        rng = np.random.RandomState(seed)
        perm_dcors.append(dcor(X, Y[rng.permutation(N)]))
        if (seed + 1) % 100 == 0:
            print(f"  perm {seed+1}/{N_PERM_DCOR}...")

    dcor_p = (np.sum(np.array(perm_dcors) >= real_dcor) + 1) / (N_PERM_DCOR + 1)
    print(f"  p = {dcor_p:.4f}{'*' if dcor_p < 0.05 else ''}")

    # Per-parameter dCor
    dcor_results = [{'comparison': 'EEG_vs_DDM_all', 'dcor': real_dcor,
                     'perm_p': dcor_p, 'n_perm': N_PERM_DCOR}]
    for i, param in enumerate(['a', 'v', 't0']):
        d = dcor(X, Y[:, i:i+1])
        dcor_results.append({'comparison': f'EEG_vs_{param}', 'dcor': d,
                             'perm_p': np.nan, 'n_perm': 0})
        print(f"  dCor(EEG, {param}) = {d:.4f}")

    pd.DataFrame(dcor_results).to_csv(OUT_DIR / "dcor_results.csv", index=False)

    # ===== CCA =====
    print(f"\n--- CCA ({N_PERM_CCA} permutations) ---")
    cca = CCA(n_components=3, max_iter=1000)
    X_c, Y_c = cca.fit_transform(X, Y)

    cc_vals = []
    for i in range(3):
        r = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        cc_vals.append(r)
        print(f"  CC{i+1}: r={r:.4f}")

    # Permutation test for CC1
    real_cc1 = cc_vals[0]
    perm_cc1 = []
    for seed in range(N_PERM_CCA):
        rng = np.random.RandomState(seed)
        try:
            cca_p = CCA(n_components=1, max_iter=500)
            Xp, Yp = cca_p.fit_transform(X, Y[rng.permutation(N)])
            perm_cc1.append(np.corrcoef(Xp[:, 0], Yp[:, 0])[0, 1])
        except Exception:
            continue
        if (seed + 1) % 1000 == 0:
            rp = (np.sum(np.array(perm_cc1) >= real_cc1) + 1) / (len(perm_cc1) + 1)
            print(f"  perm {seed+1}/{N_PERM_CCA}: running p={rp:.4f}")

    cca_p = (np.sum(np.array(perm_cc1) >= real_cc1) + 1) / (len(perm_cc1) + 1)
    print(f"  CC1 p = {cca_p:.4f}{'*' if cca_p < 0.05 else ''}")

    # Loadings: correlation of original variables with canonical variates
    loadings = []
    for cc_idx in range(3):
        for i, feat in enumerate(feat_cols):
            r = np.corrcoef(X[:, i], X_c[:, cc_idx])[0, 1]
            loadings.append({'cc': f'CC{cc_idx+1}', 'side': 'EEG',
                             'variable': feat, 'loading': r})
        for i, param in enumerate(['a', 'v', 't0']):
            r = np.corrcoef(Y[:, i], Y_c[:, cc_idx])[0, 1]
            loadings.append({'cc': f'CC{cc_idx+1}', 'side': 'DDM',
                             'variable': param, 'loading': r})

    loadings_df = pd.DataFrame(loadings)
    loadings_df.to_csv(OUT_DIR / "cca_loadings.csv", index=False)

    # Report
    write_report(real_dcor, dcor_p, cc_vals, cca_p, loadings_df,
                 feat_cols, N, OUT_DIR / "multivariate_report.md")
    print(f"\nOutput: {OUT_DIR}")


def write_report(real_dcor, dcor_p, cc_vals, cca_p, loadings_df,
                 feat_cols, N, path):
    lines = [
        "## Distance Correlation",
        "",
        f"- dCor(EEG[39], DDM[a,v,t0]) = {real_dcor:.4f}, p = {dcor_p:.4f}",
        "",
        "## CCA",
        "",
    ]
    for i, r in enumerate(cc_vals):
        p_str = f", p = {cca_p:.4f}" if i == 0 else ""
        lines.append(f"- CC{i+1}: r = {r:.4f}{p_str}")
    lines.append("")

    for cc_idx in range(2):
        cc = f'CC{cc_idx+1}'
        lines.append(f"### {cc} Loadings")
        lines.append("")
        # DDM
        ddm_load = loadings_df[(loadings_df['cc'] == cc) & (loadings_df['side'] == 'DDM')]
        lines.append("DDM:")
        for _, row in ddm_load.iterrows():
            lines.append(f"- {row['variable']}: {row['loading']:+.4f}")
        lines.append("")
        # EEG top 10
        eeg_load = loadings_df[(loadings_df['cc'] == cc) & (loadings_df['side'] == 'EEG')]
        eeg_load = eeg_load.reindex(eeg_load['loading'].abs().sort_values(ascending=False).index)
        lines.append("EEG (top 10):")
        for _, row in eeg_load.head(10).iterrows():
            lines.append(f"- {row['variable']}: {row['loading']:+.4f}")
        lines.append("")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    run()
