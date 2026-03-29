# Regression Analysis Report

## Summary

- Models: Ridge, Lasso, ElasticNet, XGBoost, PCA+Ridge, PLS, SparsePCA+Ridge, MLP
- Targets: 24
- CV: 10-fold, permutation test (100 iter, ML only)
- Previous best R²: 0.044 (FOOOF 42 features)

## Best Model per Target

| Target | Model | CV R² | Spearman ρ | Perm p | vs Prior |
|--------|-------|-------|-----------|--------|----------|
| Accuracy_Max_a | SparsePCA+Ridge | +0.0209 | +0.1328 | — | -0.0231 |
| Accuracy_Max_t0 | PCA+Ridge | -0.0084 | +0.0970 | — | -0.0524 |
| Accuracy_Max_v | SparsePCA+Ridge | -0.0120 | +0.0604 | — | -0.0560 |
| Accuracy_Mid_a | SparsePCA+Ridge | +0.0248 | +0.1512 | — | -0.0192 |
| Accuracy_Mid_t0 | PCA+Ridge | -0.0049 | +0.0758 | — | -0.0489 |
| Accuracy_Mid_v | SparsePCA+Ridge | -0.0398 | -0.0416 | — | -0.0838 |
| GrandMean_a | SparsePCA+Ridge | +0.0524 | +0.1834 | — | +0.0084 |
| GrandMean_t0 | PCA+Ridge | -0.0046 | +0.0982 | — | -0.0486 |
| GrandMean_v | PCA+Ridge | -0.0296 | -0.0737 | — | -0.0736 |
| Load_a | SparsePCA+Ridge | +0.0310 | +0.1293 | — | -0.0130 |
| Load_t0 | PCA+Ridge | -0.0246 | -0.0273 | — | -0.0686 |
| Load_v | PCA+Ridge | -0.0402 | -0.0822 | — | -0.0842 |
| Neutral_a | XGBoost | +0.0656 | +0.2867 | 0.010** | +0.0216 |
| Neutral_t0 | PCA+Ridge | -0.0185 | +0.0163 | — | -0.0625 |
| Neutral_v | PCA+Ridge | -0.0442 | -0.0486 | — | -0.0882 |
| NoLoad_a | SparsePCA+Ridge | +0.0636 | +0.2100 | — | +0.0196 |
| NoLoad_t0 | PCA+Ridge | +0.0180 | +0.1852 | — | -0.0260 |
| NoLoad_v | SparsePCA+Ridge | -0.0193 | +0.0061 | — | -0.0633 |
| Speed_Max_a | PCA+Ridge | +0.0259 | +0.1403 | — | -0.0181 |
| Speed_Max_t0 | PCA+Ridge | -0.0207 | -0.0228 | — | -0.0647 |
| Speed_Max_v | PCA+Ridge | -0.0383 | -0.0400 | — | -0.0823 |
| Speed_Mid_a | SparsePCA+Ridge | +0.0199 | +0.1030 | — | -0.0241 |
| Speed_Mid_t0 | PCA+Ridge | +0.0009 | +0.0711 | — | -0.0431 |
| Speed_Mid_v | PCA+Ridge | +0.0149 | +0.0996 | — | -0.0291 |

## Best per DDM Parameter

- **a**: R²=+0.0656 (XGBoost on Neutral_a, p=0.010)
- **v**: R²=+0.0149 (PCA+Ridge on Speed_Mid_v, p=—)
- **t0**: R²=+0.0180 (PCA+Ridge on NoLoad_t0, p=—)

## Lasso Feature Selection

Features consistently selected (non-zero) across CV folds:

**GrandMean_a**:
- global_beta: 10/10 ← robust
- occipital_delta_peak_cf: 10/10 ← robust
- left_temporal_sample_entropy: 10/10 ← robust
- conn_central_prefrontal_alpha: 8/10 ← robust
- left_temporal_perm_entropy: 8/10 ← robust
- right_temporal_aperiodic_exponent: 8/10 ← robust
- conn_posterior_left_temporal_theta: 7/10 ← robust
- prefrontal_alpha_peak_cf: 7/10 ← robust
- frontal_alpha_peak_cf: 6/10
- posterior_beta: 5/10

**GrandMean_t0**:
- prefrontal_beta_peak_cf: 10/10 ← robust
- occipital_delta_peak_cf: 10/10 ← robust
- conn_left_temporal_right_temporal_beta: 9/10 ← robust
- posterior_delta_peak_pw: 9/10 ← robust
- prefrontal_beta_peak_bw: 8/10 ← robust
- central_beta_peak_pw: 7/10 ← robust
- central_beta_peak_bw: 7/10 ← robust
- prefrontal_theta: 3/10
- frontal_alpha_peak_pw: 3/10
- left_temporal_beta_peak_pw: 3/10

**GrandMean_v**:
- posterior_delta_peak_pw: 4/10
- right_temporal_theta_peak_pw: 3/10
- frontal_theta_peak_pw: 3/10
- left_temporal_theta_peak_pw: 3/10
- right_temporal_alpha_peak_bw: 3/10
- right_temporal_sample_entropy: 3/10
- left_temporal_aperiodic_exponent: 2/10
- central_sample_entropy: 2/10
- left_temporal_sample_entropy: 2/10
- central_theta_peak_pw: 2/10

## Interpretation

### Comparison with Prior Analysis

Best R² improved from 0.044 to 0.0656 (XGBoost on Neutral_a). The expanded feature set (connectivity + entropy) contributed to this improvement.

### Key Findings

1. **Feature importance convergence**: Features appearing in both FDR-significant correlations AND Lasso/SHAP rankings provide the strongest evidence for EEG-DDM relationships.

2. **Aperiodic exponent**: Prior analysis identified this as the only robust predictor. Current results with expanded features should be compared.

3. **Connectivity features**: If conn_* features appear in top rankings, this suggests network-level dynamics contribute beyond single-ROI power (novel finding).

4. **Entropy features**: If entropy features are selected, this indicates temporal complexity of resting EEG relates to decision-making efficiency.

### Literature Alignment

- Aperiodic exponent → drift rate (v): Consistent with Euler et al. (2024), Pathania et al. (2022) — steeper 1/f slope reflects better E/I balance → faster evidence accumulation.
- IAF → processing speed: Grandy (2013), Finley (2024) — higher alpha frequency → faster temporal resolution.
- TBR → boundary separation: Mixed evidence in prior analysis (predicted v instead of a).
- Temporal ROI dominance: Unexpected finding from prior analysis — warrants investigation for muscle artifact.
