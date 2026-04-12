#!/usr/bin/env python3
"""
Analysis 1: EEG ↔ DDM Univariate Correlation (per-parameter FDR)
================================================================
39 EEG features × 3 DDM parameters (a, v, t0).
FDR correction per parameter (39 tests each).

Output:
    output/analysis1_correlation/
        correlation_results.csv
        correlation_report.md
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from shared import (BASE_DIR, load_eeg_raw, build_39_features,
                    load_ddm_grandmean, get_feature_cols)

OUT_DIR = BASE_DIR / "output" / "analysis1_correlation"


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eeg = build_39_features(load_eeg_raw())
    ddm = load_ddm_grandmean()
    feat_cols = get_feature_cols(eeg)
    merged = eeg.merge(ddm, on='subject', how='inner').dropna(subset=['a', 'v', 't0'])
    print(f"N={len(merged)}, features={len(feat_cols)}")

    # Correlation
    results = []
    for param in ['a', 'v', 't0']:
        y = merged[param].values
        for feat in feat_cols:
            x = merged[feat].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                results.append({'parameter': param, 'feature': feat,
                                'pearson_r': 0, 'spearman_rho': 0,
                                'p_value': 1, 'n': int(mask.sum())})
                continue
            r, p = pearsonr(x[mask], y[mask])
            rho = spearmanr(x[mask], y[mask]).correlation
            results.append({'parameter': param, 'feature': feat,
                            'pearson_r': r, 'spearman_rho': rho,
                            'p_value': p, 'n': int(mask.sum())})

    df = pd.DataFrame(results)

    # Per-parameter FDR
    for param in ['a', 'v', 't0']:
        mask = df['parameter'] == param
        rej, qvals, _, _ = multipletests(df.loc[mask, 'p_value'],
                                         method='fdr_bh', alpha=0.05)
        df.loc[mask, 'fdr_q'] = qvals
        df.loc[mask, 'fdr_significant'] = rej

    df.to_csv(OUT_DIR / "correlation_results.csv", index=False)

    # Report
    write_report(df, merged, OUT_DIR / "correlation_report.md")

    # Print summary
    for param in ['a', 'v', 't0']:
        sub = df[df['parameter'] == param]
        n_sig = sub['fdr_significant'].sum()
        n_trend = ((sub['fdr_q'] < 0.10) & ~sub['fdr_significant']).sum()
        n_unc = (sub['p_value'] < 0.05).sum()
        print(f"  {param}: FDR-sig={n_sig}, trend(q<0.10)={n_trend}, uncorr(p<0.05)={n_unc}")
        for _, row in sub[sub['fdr_q'] < 0.10].sort_values('fdr_q').iterrows():
            print(f"    {row['feature']:45s} r={row['pearson_r']:+.3f} q={row['fdr_q']:.4f}")

    print(f"\nOutput: {OUT_DIR}")


def write_report(df, merged, path):
    lines = []
    for param in ['a', 'v', 't0']:
        sub = df[df['parameter'] == param].sort_values('p_value')
        lines.append(f"## {param}")
        lines.append("")
        lines.append("| Feature | Pearson r | Spearman ρ | p | FDR q |")
        lines.append("|---------|-----------|-----------|---|-------|")
        for _, row in sub.iterrows():
            sig = '†' if row['fdr_q'] < 0.10 else ''
            lines.append(
                f"| {row['feature']} | {row['pearson_r']:+.3f} | "
                f"{row['spearman_rho']:+.3f} | {row['p_value']:.4f} | "
                f"{row['fdr_q']:.4f} {sig} |")
        lines.append("")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    run()
