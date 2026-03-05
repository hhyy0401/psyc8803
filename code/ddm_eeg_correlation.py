"""
Correlation analysis: ROI-based resting-state EEG features x DDM parameters (a, v).

EEG features from preprocess_v1.csv (7 ROIs x 13 features = 91 features).
DDM parameters from DDM_Scores.csv.

Three DDM aggregation methods:
  1. Load / NoLoad mean (5 conditions each)
  2. Condition-paired mean: (Load_X + NoLoad_X) / 2 for each of 5 matched pairs
  3. Grand mean across all 10 conditions

FDR correction applied per DDM variable across all EEG features.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import os

warnings.filterwarnings("ignore")

# ─── Load data ───
version = "v2"   #####manually change this

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ddm = pd.read_csv(os.path.join(base_dir, "DDM_Scores.csv"))
eeg = pd.read_csv(os.path.join(base_dir, f"preprocess_{version}.csv"))

# ─── Preprocessing ───
eeg_features = [c for c in eeg.columns if c != "subject"]

eeg.rename(columns={"subject": "Subject"}, inplace=True)
eeg_clean = eeg.dropna(subset=eeg_features).copy()

ddm["a"] = pd.to_numeric(ddm["a"], errors="coerce")
ddm["v"] = pd.to_numeric(ddm["v"], errors="coerce")
ddm_clean = ddm.dropna(subset=["a", "v"]).copy()

common_subjects = set(eeg_clean["Subject"]) & set(ddm_clean["Subject"])
print(f"Common subjects: {len(common_subjects)}")

eeg_sub = eeg_clean[eeg_clean["Subject"].isin(common_subjects)].set_index("Subject")
ddm_sub = ddm_clean[ddm_clean["Subject"].isin(common_subjects)].copy()


# ─── DDM aggregation ───
def aggregate_overall(df):
    """Grand mean of a, v across all 10 conditions."""
    return df.groupby("Subject")[["a", "v"]].mean()


def aggregate_load_noload(df):
    """Separate means for Load (5) and NoLoad (5) conditions."""
    load = df[df["Load_Condition"] == "Load"].groupby("Subject")[["a", "v"]].mean()
    load.columns = ["Load_a", "Load_v"]
    noload = df[df["Load_Condition"] == "NoLoad"].groupby("Subject")[["a", "v"]].mean()
    noload.columns = ["NoLoad_a", "NoLoad_v"]
    return load.join(noload, how="inner")


def aggregate_condition_paired_mean(df):
    """Mean of (Load_X + NoLoad_X) / 2 for each of the 5 matched pairs."""
    conditions = ["Accuracy_Max", "Accuracy_Mid", "Neutral", "Speed_Max", "Speed_Mid"]
    result = pd.DataFrame()
    for cond in conditions:
        load_cond = df[df["Detailed_Condition"] == f"Load_{cond}"].set_index("Subject")[["a", "v"]]
        noload_cond = df[df["Detailed_Condition"] == f"NoLoad_{cond}"].set_index("Subject")[["a", "v"]]
        common_idx = load_cond.index.intersection(noload_cond.index)
        paired_mean = (load_cond.loc[common_idx] + noload_cond.loc[common_idx]) / 2
        paired_mean.columns = [f"{cond}_a", f"{cond}_v"]
        if result.empty:
            result = paired_mean
        else:
            result = result.join(paired_mean, how="inner")
    return result


# ─── Correlation with per-variable FDR ───
def run_correlations(ddm_df, eeg_df, eeg_features, ddm_cols, method_name):
    common = ddm_df.index.intersection(eeg_df.index)
    ddm_aligned = ddm_df.loc[common]
    eeg_aligned = eeg_df.loc[common, eeg_features]

    all_results = []
    for ddm_col in ddm_cols:
        var_results = []
        for eeg_col in eeg_features:
            x = eeg_aligned[eeg_col].values
            y = ddm_aligned[ddm_col].values

            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean, y_clean = x[mask], y[mask]
            if len(x_clean) < 10:
                continue

            r_pearson, p_pearson = stats.pearsonr(x_clean, y_clean)
            r_spearman, p_spearman = stats.spearmanr(x_clean, y_clean)

            var_results.append({
                "Method": method_name,
                "DDM_Variable": ddm_col,
                "EEG_Feature": eeg_col,
                "N": len(x_clean),
                "Pearson_r": round(r_pearson, 4),
                "Pearson_p": p_pearson,
                "Spearman_rho": round(r_spearman, 4),
                "Spearman_p": p_spearman,
            })

        # FDR per DDM variable
        if var_results:
            df_var = pd.DataFrame(var_results)
            _, fdr_p, _, _ = multipletests(df_var["Pearson_p"], method="fdr_bh")
            _, fdr_s, _, _ = multipletests(df_var["Spearman_p"], method="fdr_bh")
            df_var["Pearson_FDR"] = fdr_p
            df_var["Spearman_FDR"] = fdr_s
            all_results.append(df_var)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


# ─── Run analysis ───
print("\n" + "=" * 80)
print("DDM Scores x ROI-based Resting EEG Features: Correlation Analysis")
print(f"EEG features: {len(eeg_features)} (7 ROIs x 13 features)")
print("FDR correction applied per DDM variable")
print("=" * 80)

# Method 1: Load/NoLoad mean
ddm_loadnoload = aggregate_load_noload(ddm_sub)
res1 = run_correlations(ddm_loadnoload, eeg_sub, eeg_features,
                        ["Load_a", "Load_v", "NoLoad_a", "NoLoad_v"], "Load/NoLoad")

# Method 2: Condition-paired mean
ddm_paired = aggregate_condition_paired_mean(ddm_sub)
paired_cols = list(ddm_paired.columns)
res2 = run_correlations(ddm_paired, eeg_sub, eeg_features, paired_cols, "PairedMean")

# Method 3: Grand mean
ddm_overall = aggregate_overall(ddm_sub)
res3 = run_correlations(ddm_overall, eeg_sub, eeg_features, ["a", "v"], "GrandMean")

all_results = pd.concat([res1, res2, res3], ignore_index=True)


# ─── Print results ───
def print_results(df, title):
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")

    if df.empty:
        print("  No results")
        return

    for ddm_var in df["DDM_Variable"].unique():
        var_df = df[df["DDM_Variable"] == ddm_var].sort_values("Pearson_p")
        sig_df = var_df[(var_df["Pearson_p"] < 0.05) | (var_df["Spearman_p"] < 0.05)]
        print(f"\n  [{ddm_var}] ({len(sig_df)}/{len(var_df)} nominally significant)")
        # Show top 10 by Pearson p
        for _, row in var_df.head(10).iterrows():
            fdr_p = " *FDR" if row["Pearson_FDR"] < 0.05 else ""
            fdr_s = " *FDR" if row["Spearman_FDR"] < 0.05 else ""
            p_sig = "***" if row["Pearson_p"] < 0.001 else ("**" if row["Pearson_p"] < 0.01 else ("*" if row["Pearson_p"] < 0.05 else "  "))
            s_sig = "***" if row["Spearman_p"] < 0.001 else ("**" if row["Spearman_p"] < 0.01 else ("*" if row["Spearman_p"] < 0.05 else "  "))

            print(f"    {row['EEG_Feature']:45s} | "
                  f"r={row['Pearson_r']:+.4f} p={row['Pearson_p']:.4f}{p_sig} FDR={row['Pearson_FDR']:.4f}{fdr_p} | "
                  f"rho={row['Spearman_rho']:+.4f} p={row['Spearman_p']:.4f}{s_sig} FDR={row['Spearman_FDR']:.4f}{fdr_s} | "
                  f"N={row['N']}")


print_results(res1, "Method 1: Load / NoLoad mean (5 conditions each)")
print_results(res2, "Method 2: Condition-paired mean (Load_X + NoLoad_X) / 2")
print_results(res3, "Method 3: Grand mean (all 10 conditions)")

# ─── Save ───
output_path = os.path.join(base_dir, "analysis", "DDM_EEG_Correlation_Results.csv")
all_results.to_csv(output_path, index=False)
print(f"\n  Results saved to: {output_path}")


# ─── Generate markdown report ───
def generate_markdown_report(all_results):
    lines = []
    lines.append("# DDM x Resting EEG Correlation Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **EEG Features**: ROI-based band power from `preprocess_{version}.csv` (7 ROIs x 13 features = 91)")
    lines.append("- **DDM Parameters**: Boundary separation (a) and drift rate (v)")
    lines.append("- **Correction**: FDR (Benjamini-Hochberg) per DDM variable")
    lines.append(f"- **N subjects**: {all_results['N'].max() if not all_results.empty else 'N/A'}")
    lines.append("")

    lines.append("## ROI Definitions")
    lines.append("")
    lines.append("| ROI | Channels |")
    lines.append("|-----|----------|")
    rois = {
        "Frontal": "E3, E6, E8, E9",
        "Posterior": "E34, E31, E40, E33, E38, E36",
        "Central": "E16, E7, E4, E54, E51, E41, E21",
        "Left Temporal": "E22, E24, E25, E30",
        "Right Temporal": "E52, E48, E45, E44",
        "Occipital": "E36, E37, E39",
        "Prefrontal": "E1, E17, E2, E11, E5, E10",
    }
    for roi, ch in rois.items():
        lines.append(f"| {roi} | {ch} |")
    lines.append("")

    lines.append("## EEG Features per ROI")
    lines.append("")
    lines.append("- Absolute band power: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-40 Hz)")
    lines.append("- Relative band power: each band / total power")
    lines.append("- Theta/beta ratio")
    lines.append("- Peak alpha frequency (PAF)")
    lines.append("- Spectral entropy")
    lines.append("")

    for method_name in ["Load/NoLoad", "PairedMean", "GrandMean"]:
        method_df = all_results[all_results["Method"] == method_name]
        if method_df.empty:
            continue

        title_map = {
            "Load/NoLoad": "Method 1: Load / NoLoad Mean",
            "PairedMean": "Method 2: Condition-Paired Mean",
            "GrandMean": "Method 3: Grand Mean",
        }
        lines.append(f"## {title_map[method_name]}")
        lines.append("")

        for ddm_var in method_df["DDM_Variable"].unique():
            var_df = method_df[method_df["DDM_Variable"] == ddm_var].sort_values("Pearson_p")
            fdr_sig = var_df[var_df["Pearson_FDR"] < 0.05]
            nom_sig = var_df[var_df["Pearson_p"] < 0.05]

            lines.append(f"### {ddm_var}")
            lines.append("")
            lines.append(f"FDR-significant: **{len(fdr_sig)}** / {len(var_df)} features")
            lines.append(f"Nominally significant (p < .05): **{len(nom_sig)}** / {len(var_df)} features")
            lines.append("")

            # Show top results
            show_df = var_df.head(15)
            if not show_df.empty:
                lines.append("| EEG Feature | Pearson r | p | FDR p | Spearman rho | p | FDR p | N |")
                lines.append("|-------------|-----------|---|-------|--------------|---|-------|---|")
                for _, row in show_df.iterrows():
                    p_star = "***" if row["Pearson_p"] < 0.001 else ("**" if row["Pearson_p"] < 0.01 else ("*" if row["Pearson_p"] < 0.05 else ""))
                    lines.append(
                        f"| {row['EEG_Feature']} "
                        f"| {row['Pearson_r']:+.4f} "
                        f"| {row['Pearson_p']:.4f}{p_star} "
                        f"| {row['Pearson_FDR']:.4f} "
                        f"| {row['Spearman_rho']:+.4f} "
                        f"| {row['Spearman_p']:.4f} "
                        f"| {row['Spearman_FDR']:.4f} "
                        f"| {row['N']} |"
                    )
                lines.append("")

    # Summary of FDR-significant results
    fdr_all = all_results[all_results["Pearson_FDR"] < 0.05]
    lines.append("## Summary: FDR-Significant Results")
    lines.append("")
    if fdr_all.empty:
        lines.append("No correlations survived FDR correction (q < .05).")
    else:
        lines.append(f"**{len(fdr_all)}** correlations survived FDR correction:")
        lines.append("")
        lines.append("| Method | DDM Var | EEG Feature | r | p | FDR p |")
        lines.append("|--------|---------|-------------|---|---|-------|")
        for _, row in fdr_all.sort_values("Pearson_FDR").iterrows():
            lines.append(
                f"| {row['Method']} | {row['DDM_Variable']} | {row['EEG_Feature']} "
                f"| {row['Pearson_r']:+.4f} | {row['Pearson_p']:.4f} | {row['Pearson_FDR']:.4f} |"
            )
    lines.append("")

    return "\n".join(lines)


report = generate_markdown_report(all_results)
report_path = os.path.join(base_dir, "analysis", f"corr_results_{version}.md")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    f.write(report)
print(f"  Report saved to: {report_path}")
