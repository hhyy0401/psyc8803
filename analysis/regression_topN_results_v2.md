# DDM Regression with FDR-Based Feature Selection

## Overview

- **Feature selection**: Top-N FDR-significant features from correlation analysis
- **Feature sets differ for a vs v targets** (selected separately based on correlation patterns)
- **Models**: Ridge, Lasso, ElasticNet, XGBoost, SVR
- **Evaluation**: 10-fold CV R², MAE, permutation test (200 iter; skipped if R² <= 0)
- **Top-N values tested**: [8]
- **N subjects**: 184

## Top N = 8

### Best Model per Target

| Target | Model | CV R² | MAE | Perm p | Features Used |
|--------|-------|--------|-----|--------|---------------|
| Load_a | Ridge | +0.0300 | 0.3359 | 0.0149 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| Load_v | SVR | +0.0145 | 0.6345 | 0.0398 * | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| NoLoad_a | Ridge | +0.0327 | 0.2326 | 0.0100 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| NoLoad_v | Lasso | +0.0402 | 0.6856 | 0.0100 * | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| Accuracy_Max_a | Ridge | +0.0240 | 0.3224 | 0.0348 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| Accuracy_Max_v | SVR | +0.0139 | 0.7116 | 0.0448 * | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| Accuracy_Mid_a | Ridge | +0.0137 | 0.3419 | 0.0448 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| Accuracy_Mid_v | Ridge | -0.0009 | 0.8157 | 1.0000  | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| Neutral_a | Ridge | +0.0319 | 0.2958 | 0.0100 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| Neutral_v | SVR | +0.0005 | 0.6881 | 0.0697  | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| Speed_Max_a | Ridge | +0.0135 | 0.3416 | 0.0448 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| Speed_Max_v | SVR | -0.0035 | 0.8209 | 1.0000  | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| Speed_Mid_a | Ridge | +0.0192 | 0.3047 | 0.0299 * | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| Speed_Mid_v | Ridge | +0.0379 | 0.7800 | 0.0149 * | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
| a | Ridge | +0.0385 | 0.2642 | 0.0050 ** | left_temporal_abs_gamma, right_temporal_abs_gamma, left_temporal_rel_gamma, prefrontal_abs_beta, right_temporal_abs_beta, central_abs_gamma, occipital_rel_theta, left_temporal_abs_beta |
| v | Lasso | +0.0249 | 0.6090 | 0.0100 * | central_theta_beta_ratio, right_temporal_spectral_entropy, right_temporal_theta_beta_ratio, right_temporal_rel_beta, right_temporal_rel_gamma, posterior_rel_beta, central_rel_beta, central_rel_gamma |
