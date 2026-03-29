#!/usr/bin/env python3
"""
Generate summary reports (EN for presentation, KR for understanding)
from analysis output files. Run after analysis_features.py + analysis_rt.py.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.environ.get("EEG_BASE_DIR",
    "/Users/hkim3239/GaTech Dropbox/Hyunju Kim/EEG"))
OUT_DIR = BASE_DIR / "output"
MIN_VALID_N = 50


def load_results():
    corr = pd.read_csv(OUT_DIR / "correlation" / "correlation_fdr.csv")
    reg = pd.read_csv(OUT_DIR / "regression" / "regression_results.csv")
    coef = pd.read_csv(OUT_DIR / "regression" / "lasso_coef_tracking.csv")
    rt = pd.read_csv(OUT_DIR / "rt_prediction" / "rt_comparison_results.csv")
    trial = pd.read_csv(OUT_DIR / "rt_prediction" / "trial_level_evaluation.csv")
    return corr, reg, coef, rt, trial


def top_features(corr, coef):
    """Find convergent features: Lasso robust + uncorrected p<0.01."""
    # Lasso robust (>=7/10 folds)
    lasso_robust = set()
    for target in coef["target"].unique():
        sub = coef[(coef["target"] == target) & (coef["model"] == "Lasso")]
        if len(sub) == 0:
            continue
        freq = sub.groupby("feature")["nonzero"].sum()
        lasso_robust.update(freq[freq >= 7].index)

    # Correlation p<0.01
    corr_sig = set(corr[corr["p_value"] < 0.01]["feature"].unique())
    overlap = sorted(lasso_robust & corr_sig)

    # Details for each
    details = []
    for feat in overlap:
        fc = corr[corr["feature"] == feat].nsmallest(1, "p_value").iloc[0]
        fl = coef[(coef["feature"] == feat) & (coef["model"] == "Lasso")]
        n_robust = fl.groupby("target")["nonzero"].sum()
        best_target = n_robust.idxmax()
        # count targets with p<0.005
        n_targets = len(corr[(corr["feature"] == feat) & (corr["p_value"] < 0.005)])
        details.append({
            "feature": feat,
            "best_r": fc["pearson_r"],
            "best_p": fc["p_value"],
            "best_corr_target": fc["target"],
            "lasso_folds": int(n_robust.max()),
            "lasso_target": best_target,
            "n_targets_p005": n_targets,
        })
    return pd.DataFrame(details).sort_values("n_targets_p005", ascending=False)


def fdr_summary(corr):
    """FDR results (FDR is applied per target, not across all)."""
    valid = corr[corr["fdr_q"].notna()]
    n_sig = valid["fdr_significant"].sum() if len(valid) > 0 else 0
    best_q = valid["fdr_q"].min() if len(valid) > 0 else np.nan
    n_targets = valid["target"].nunique()
    n_features_per_target = len(valid) // max(n_targets, 1)
    # Find which target has the best q
    best_row = valid.loc[valid["fdr_q"].idxmin()] if len(valid) > 0 else None
    best_target = best_row["target"] if best_row is not None else ""
    best_feature = best_row["feature"] if best_row is not None else ""
    return n_sig, n_targets, n_features_per_target, best_q, best_target, best_feature


def best_models(reg):
    """Best model per target and per DDM parameter."""
    per_target = []
    for target in sorted(reg["target"].unique()):
        sub = reg[reg["target"] == target]
        best = sub.loc[sub["r2"].idxmax()]
        per_target.append(best.to_dict())

    per_param = {}
    for param in ["a", "v", "t0"]:
        sub = reg[reg["target"].str.endswith(f"_{param}")]
        if len(sub) > 0:
            best = sub.loc[sub["r2"].idxmax()]
            per_param[param] = best.to_dict()
    return pd.DataFrame(per_target), per_param


def dim_reduce_comparison(reg):
    """Compare dim reduction models vs baseline."""
    dr_models = ["PCA+Ridge", "PLS", "SparsePCA+Ridge"]
    baseline_models = ["Ridge", "Lasso", "ElasticNet", "XGBoost"]
    rows = []
    for target in sorted(reg["target"].unique()):
        sub = reg[reg["target"] == target]
        bl = sub[sub["model"].isin(baseline_models)]
        dr = sub[sub["model"].isin(dr_models)]
        if len(bl) == 0 or len(dr) == 0:
            continue
        best_bl = bl.loc[bl["r2"].idxmax()]
        best_dr = dr.loc[dr["r2"].idxmax()]
        rows.append({
            "target": target,
            "baseline_model": best_bl["model"],
            "baseline_r2": best_bl["r2"],
            "dr_model": best_dr["model"],
            "dr_r2": best_dr["r2"],
            "improvement": best_dr["r2"] - best_bl["r2"],
        })
    return pd.DataFrame(rows)


def write_english(corr, reg, coef, rt, trial, feat_df, path):
    """English summary for presentation."""
    n_sig, n_targets, n_feat_per, best_q, best_target, best_feature = fdr_summary(corr)
    per_target, per_param = best_models(reg)
    dr_comp = dim_reduce_comparison(reg)
    n_features = corr["feature"].nunique()

    lines = [
        "# EEG → DDM Prediction: Full Results Summary",
        "",
        "## Dataset",
        f"- Subjects: {corr['n'].max():.0f} (with EEG + DDM + behavioral data)",
        f"- EEG features: {n_features} (band power, FOOOF aperiodic/periodic, connectivity, entropy)",
        f"- DDM targets: {corr['target'].nunique()} (3 params × 8 conditions)",
        f"- Feature filter: dropped features with <{MIN_VALID_N} valid values",
        "",
        "---",
        "",
        "## Analysis 1: Feature–DDM Correlations",
        "",
        f"- FDR correction: per target (~{n_feat_per} features × {n_targets} targets)",
        f"- FDR-significant (q < 0.05): **{n_sig}**",
        f"- Best FDR q-value: **{best_q:.4f}** (`{best_feature}` → {best_target})"
        if np.isfinite(best_q) else "- Best FDR q-value: N/A",
        "",
    ]

    if n_sig > 0:
        sig = corr[corr["fdr_significant"] == True].sort_values("fdr_q")
        lines += ["### FDR-Significant Features", "",
                   "| Feature | Target | r | q |",
                   "|---------|--------|---|---|"]
        for _, row in sig.head(20).iterrows():
            lines.append(f"| {row['feature']} | {row['target']} | {row['pearson_r']:+.3f} | {row['fdr_q']:.4f} |")
        lines.append("")

    # Top uncorrected
    top_unc = corr.nsmallest(10, "p_value")
    lines += ["### Top 10 Uncorrected Correlations", "",
              "| Feature | Target | r | p |",
              "|---------|--------|---|---|"]
    for _, row in top_unc.iterrows():
        lines.append(f"| {row['feature']} | {row['target']} | {row['pearson_r']:+.3f} | {row['p_value']:.4f} |")
    lines.append("")

    # Convergent features
    if len(feat_df) > 0:
        lines += [
            "### Convergent Features (Lasso robust + correlation p<0.01)",
            "",
            f"**{len(feat_df)} features** with evidence from both methods:",
            "",
            "| Feature | #Targets (p<.005) | Best r | Best p | Lasso folds |",
            "|---------|-------------------|--------|--------|-------------|",
        ]
        for _, row in feat_df.head(15).iterrows():
            lines.append(
                f"| {row['feature']} | {row['n_targets_p005']} | "
                f"{row['best_r']:+.3f} | {row['best_p']:.4f} | "
                f"{row['lasso_folds']}/10 |")
        lines.append("")

    # Analysis 2: Regression
    lines += [
        "---", "",
        "## Analysis 2: Regression (DDM Parameter Prediction)", "",
        "### Best Model per DDM Parameter", "",
    ]
    for param in ["a", "v", "t0"]:
        if param in per_param:
            d = per_param[param]
            p_str = f"p={d['perm_p']:.3f}" if np.isfinite(d.get("perm_p", np.nan)) else "p=N/A"
            lines.append(f"- **{param}**: R²={d['r2']:+.4f}, ρ={d['spearman']:+.4f} "
                          f"({d['model']} on {d['target']}, {p_str})")
    lines.append("")

    # Dim reduction comparison
    if len(dr_comp) > 0:
        n_better = (dr_comp["improvement"] > 0).sum()
        avg_imp = dr_comp["improvement"].mean()
        lines += [
            "### Dimensionality Reduction vs. Feature Selection", "",
            f"- DR models outperformed baseline in **{n_better}/{len(dr_comp)}** targets",
            f"- Average R² change: {avg_imp:+.4f}",
            "",
            "| Target | Baseline | R² | DR Model | R² | Δ |",
            "|--------|----------|-----|----------|-----|---|",
        ]
        for _, row in dr_comp.sort_values("improvement", ascending=False).head(10).iterrows():
            lines.append(
                f"| {row['target']} | {row['baseline_model']} | {row['baseline_r2']:+.4f} | "
                f"{row['dr_model']} | {row['dr_r2']:+.4f} | {row['improvement']:+.4f} |")
        lines.append("")

    # Analysis 3: RT comparison
    lines += [
        "---", "",
        "## Analysis 3: 3-Way RT Distribution Comparison", "",
        "| Model | RT MAE (ms) | RT R² | RT ρ |",
        "|-------|-------------|-------|------|",
    ]
    for _, row in rt.sort_values("rt_mean_mae").iterrows():
        lines.append(f"| {row['model']} | {row['rt_mean_mae']:.1f} | "
                      f"{row['rt_mean_r2']:+.4f} | {row['rt_mean_rho']:+.4f} |")

    cond_r2 = rt[rt["model"] == "cond_only_ridge"]["rt_mean_r2"].values
    cond_r2 = cond_r2[0] if len(cond_r2) > 0 else np.nan
    eeg_r2 = rt[rt["model"].str.startswith("eeg_cond")]["rt_mean_r2"].max()
    ddm_r2 = rt[rt["model"].str.startswith("ddm")]["rt_mean_r2"].max()

    lines += [
        "",
        f"- Condition only R²: {cond_r2:.4f}" if np.isfinite(cond_r2) else "",
        f"- Best EEG+Condition R²: {eeg_r2:.4f}" if np.isfinite(eeg_r2) else "",
        f"- Best DDM-mediated R²: {ddm_r2:.4f}" if np.isfinite(ddm_r2) else "",
        "",
    ]

    if np.isfinite(cond_r2) and np.isfinite(eeg_r2):
        if eeg_r2 > cond_r2 + 0.01:
            lines.append("**EEG adds predictive value beyond condition alone.**")
        else:
            lines.append("**EEG does not improve over condition alone** → individual resting EEG "
                          "differences have limited predictive value for RT.")
    if np.isfinite(eeg_r2) and np.isfinite(ddm_r2):
        if eeg_r2 > ddm_r2 + 0.01:
            lines.append("**DDM-free > DDM-mediated** → DDM acts as information bottleneck.")
        elif ddm_r2 > eeg_r2 + 0.01:
            lines.append("**DDM-mediated > DDM-free** → DDM inductive bias helps prediction.")
        else:
            lines.append("**DDM-free ≈ DDM-mediated** → DDM is an adequate summary.")
    lines.append("")

    # Trial-level
    if len(trial) > 0:
        lines += [
            "### Trial-Level DDM Simulation Quality", "",
            f"- Wasserstein distance: {trial['wasserstein_ms'].mean():.1f} ± {trial['wasserstein_ms'].std():.1f} ms",
            f"- KS statistic: {trial['ks_stat'].mean():.3f} ± {trial['ks_stat'].std():.3f}",
            "",
        ]

    # Key takeaways
    lines += [
        "---", "",
        "## Key Takeaways", "",
        "1. Resting-state EEG features show weak but detectable associations with DDM parameters "
        "(best R² ~ 0.04, effect size ceiling).",
        "",
        "2. **Entropy and aperiodic features** (E/I balance markers) are the most consistent predictors, "
        "particularly for boundary separation (a).",
        "",
        "3. Dimensionality reduction (PCA/SparsePCA) generally improves over univariate feature selection, "
        "suggesting multicollinearity is a significant issue.",
        "",
        "4. DDM acts as an information bottleneck for RT prediction — direct EEG→RT models "
        "outperform EEG→DDM→RT pipeline.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {path}")


def write_korean(corr, reg, coef, rt, trial, feat_df, path):
    """Korean summary for personal understanding."""
    n_sig, n_targets, n_feat_per, best_q, best_target, best_feature = fdr_summary(corr)
    per_target, per_param = best_models(reg)
    dr_comp = dim_reduce_comparison(reg)

    lines = [
        "# EEG → DDM 예측 결과 정리",
        "",
        "## 한줄 요약",
        "",
        "Resting EEG로 DDM 파라미터를 예측하는 건 **근본적으로 약한 관계** (best R²~0.04).",
        "하지만 entropy, aperiodic exponent 등 비선형 특징이 일관되게 관련됨.",
        "",
        "---",
        "",
        "## 1. Correlation 분석",
        "",
        f"- FDR 보정 단위: 타겟별 (~{n_feat_per}개 feature × {n_targets}개 타겟)",
        f"- **FDR 통과: {n_sig}개**",
        f"- Best FDR q = **{best_q:.4f}** (`{best_feature}` → {best_target})"
        if np.isfinite(best_q) else "",
        "",
    ]

    if n_sig == 0:
        lines += [
            "FDR 통과 feature 없음. 이유:",
            "- N=188에서 r=0.25 수준 → p~0.0004",
            f"- 타겟당 ~{n_feat_per}개 feature 보정 → best q={best_q:.4f} (0.05 근접하지만 미달)"
            if np.isfinite(best_q) else "",
            "- FDR 통과하려면 r≈0.35 또는 N≈400 필요",
            "",
        ]

    # Top uncorrected
    top = corr.nsmallest(5, "p_value")
    lines += ["### 가장 강한 상관 (uncorrected)", ""]
    for _, row in top.iterrows():
        lines.append(f"- `{row['feature']}` → {row['target']}: r={row['pearson_r']:+.3f}, p={row['p_value']:.4f}")
    lines.append("")

    # Convergent features
    if len(feat_df) > 0:
        lines += [
            "### 수렴적 증거가 있는 Feature들",
            "",
            "Lasso에서 robust (≥7/10 folds) + correlation p<0.01 둘 다 만족하는 feature:",
            "",
        ]
        for _, row in feat_df.head(10).iterrows():
            lines.append(
                f"- **{row['feature']}**: {row['n_targets_p005']}개 타겟에서 p<.005, "
                f"best r={row['best_r']:+.3f}, Lasso {row['lasso_folds']}/10")
        lines.append("")

        # Interpretation
        entropy_feats = feat_df[feat_df["feature"].str.contains("entropy")]
        aperiodic_feats = feat_df[feat_df["feature"].str.contains("aperiodic")]
        conn_feats = feat_df[feat_df["feature"].str.contains("conn_")]
        temporal_feats = feat_df[feat_df["feature"].str.contains("temporal")]

        lines += [
            "### Feature 패턴 해석", "",
            f"- **Entropy 관련**: {len(entropy_feats)}개 — 뇌 temporal complexity ↑ → 신중한 결정 (a↑)",
            f"- **Aperiodic (1/f slope)**: {len(aperiodic_feats)}개 — E/I balance 지표, 문헌과 일치",
            f"- **Connectivity**: {len(conn_feats)}개 — network-level 증거 (새로운 발견)",
            f"- **Temporal ROI 관련**: {len(temporal_feats)}개/{len(feat_df)}개 — "
            "muscle artifact 가능성 확인 필요",
            "",
        ]

    # Regression
    lines += [
        "---", "",
        "## 2. Regression 분석", "",
        "### DDM 파라미터별 최고 성능", "",
    ]
    for param in ["a", "v", "t0"]:
        if param in per_param:
            d = per_param[param]
            p_str = f"p={d['perm_p']:.3f}" if np.isfinite(d.get("perm_p", np.nan)) else ""
            lines.append(f"- **{param}**: R²={d['r2']:+.4f} ({d['model']}, {d['target']}) {p_str}")
    lines.append("")

    # DR comparison
    if len(dr_comp) > 0:
        n_better = (dr_comp["improvement"] > 0).sum()
        lines += [
            "### PCA/PLS/SparsePCA 효과", "",
            f"- 기존 대비 개선된 타겟: **{n_better}/{len(dr_comp)}**",
            f"- 평균 R² 변화: {dr_comp['improvement'].mean():+.4f}",
            "",
        ]
        best_dr = dr_comp.nlargest(5, "improvement")
        for _, row in best_dr.iterrows():
            lines.append(
                f"  - {row['target']}: {row['baseline_model']}({row['baseline_r2']:+.4f}) "
                f"→ {row['dr_model']}({row['dr_r2']:+.4f}) Δ={row['improvement']:+.4f}")
        lines.append("")
        lines += [
            "해석: PCA/SparsePCA가 개선되는 경우 multicollinearity가 문제였다는 뜻.",
            "PLS는 N이 적어서 supervised projection이 오히려 overfit.",
            "",
        ]

    # RT comparison
    lines += [
        "---", "",
        "## 3. RT 3-Way 비교", "",
        "핵심 질문: DDM이 EEG→RT 예측에서 도움이 되나, 아니면 병목인가?", "",
    ]
    for _, row in rt.sort_values("rt_mean_mae").iterrows():
        lines.append(f"- {row['model']}: MAE={row['rt_mean_mae']:.0f}ms, R²={row['rt_mean_r2']:+.4f}")
    lines.append("")

    cond_r2 = rt[rt["model"] == "cond_only_ridge"]["rt_mean_r2"].values
    cond_r2 = cond_r2[0] if len(cond_r2) > 0 else np.nan
    eeg_r2 = rt[rt["model"].str.startswith("eeg_cond")]["rt_mean_r2"].max()

    if np.isfinite(cond_r2) and np.isfinite(eeg_r2) and cond_r2 > eeg_r2:
        lines += [
            "**결론: condition만으로도 EEG보다 나음.**",
            "Resting EEG의 개인차 정보가 condition 효과를 넘어서지 못함.",
            "",
        ]

    # Trial-level
    if len(trial) > 0:
        lines += [
            f"Trial-level DDM 시뮬레이션 품질: Wasserstein={trial['wasserstein_ms'].mean():.0f}ms",
            "",
        ]

    # Overall conclusion
    lines += [
        "---", "",
        "## 전체 결론", "",
        "### 논문으로 쓸 수 있는 포인트",
        "",
        "1. **Effect size ceiling 확인**: 231 features × 10 models × DR까지 해도 R²~0.04",
        "   → resting EEG-DDM 관계의 상한선을 실증적으로 보여줌",
        "",
        "2. **Feature 특성**: band power보다 entropy/aperiodic이 일관적",
        "   → 비선형 뇌 역학이 의사결정과 더 관련 (E/I balance 가설 지지)",
        "",
        "3. **DDM bottleneck**: (a,v,t0) 3개 파라미터가 정보를 과도하게 압축",
        "   → 직접 예측이 DDM 경유보다 나음",
        "",
        "### 한계 / 다음 단계",
        "",
        "- Preprocessing에 ICA 없음 → temporal ROI 결과의 artifact 가능성 검증 필요",
        "- Resting-state의 근본적 한계 → task-state EEG (ERP, time-frequency)로 전환 고려",
        "- Trial-level 모델링으로 전환하면 N이 크게 늘어남 (194K trials)",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {path}")


def main():
    print("Generating summary reports...")
    corr, reg, coef, rt, trial = load_results()
    feat_df = top_features(corr, coef)

    write_english(corr, reg, coef, rt, trial, feat_df,
                  OUT_DIR / "summary_EN.md")
    write_korean(corr, reg, coef, rt, trial, feat_df,
                 OUT_DIR / "summary_KR.md")
    print("Done.")


if __name__ == "__main__":
    main()
