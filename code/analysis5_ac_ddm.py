#!/usr/bin/env python3
"""
Analysis 5: AC ↔ DDM Correlation
=================================
Tests the link between Attentional Control (AC) and DDM parameters.
Establishes the indirect pathway: EEG → AC → DDM.

Output:
    output/analysis5_ac_ddm/
        ac_ddm_correlation.csv
        ac_ddm_report.md
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from shared import BASE_DIR, load_ddm_grandmean, load_composites

OUT_DIR = BASE_DIR / "output" / "analysis5_ac_ddm"
COMPOSITE_COLS = ['WMC', 'gF', 'AC', 'SuS_AC']


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    composites = load_composites()
    ddm = load_ddm_grandmean()
    merged = composites.merge(ddm, on='subject', how='inner')
    merged = merged.dropna(subset=['a', 'v', 't0'])
    print(f"N={len(merged)}")

    results = []
    for comp in COMPOSITE_COLS:
        valid = merged[comp].notna()
        m = merged[valid]
        x = m[comp].values
        for param in ['a', 'v', 't0']:
            y = m[param].values
            r, p = pearsonr(x, y)
            rho = spearmanr(x, y).correlation
            _, sp = spearmanr(x, y)
            results.append({
                'composite': comp, 'ddm_param': param,
                'pearson_r': r, 'pearson_p': p,
                'spearman_rho': rho, 'n': len(m),
            })
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {comp:8s} ↔ {param}: r={r:+.3f} p={p:.6f} {sig}")
        print()

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "ac_ddm_correlation.csv", index=False)

    # Report
    write_report(df, len(merged), OUT_DIR / "ac_ddm_report.md")
    print(f"Output: {OUT_DIR}")


def write_report(df, N, path):
    lines = [
        f"N = {N}",
        "",
        "## All Composite ↔ DDM Correlations",
        "",
        "| Composite | DDM | Pearson r | p |",
        "|-----------|-----|-----------|---|",
    ]
    for _, row in df.sort_values(['composite', 'ddm_param']).iterrows():
        sig = '***' if row['pearson_p'] < 0.001 else ('**' if row['pearson_p'] < 0.01 else ('*' if row['pearson_p'] < 0.05 else ''))
        lines.append(
            f"| {row['composite']} | {row['ddm_param']} | "
            f"{row['pearson_r']:+.3f} | {row['pearson_p']:.6f} {sig} |")
    lines.append("")

    # Highlight AC
    ac = df[df['composite'] == 'AC']
    lines.append("## AC (Attentional Control) — Key Results")
    lines.append("")
    for _, row in ac.iterrows():
        lines.append(f"- AC ↔ {row['ddm_param']}: r={row['pearson_r']:+.3f}, p={row['pearson_p']:.6f}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    run()
