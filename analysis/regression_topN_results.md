# DDM Regression with FDR-Based Feature Selection

## Overview

- **Feature selection**: Top-N FDR-significant features from correlation analysis
- **Feature sets differ for a vs v targets** (selected separately based on correlation patterns)
- **Models**: Ridge, Lasso, ElasticNet, XGBoost, SVR
- **Evaluation**: 10-fold CV R², MAE, permutation test (200 iter; skipped if R² <= 0)
- **Top-N values tested**: [10]
- **N subjects**: 184

## Top N = 10

### Best Model per Target

| Target | Model | CV R² | MAE | Perm p | Features Used |
|--------|-------|--------|-----|--------|---------------|
| Load_a | ElasticNet | +0.1350 | 0.3277 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| Load_v | ElasticNet | +0.0658 | 0.6417 | 0.0050 ** | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| NoLoad_a | ElasticNet | +0.1375 | 0.2285 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| NoLoad_v | Lasso | +0.0689 | 0.6823 | 0.0050 ** | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| Accuracy_Max_a | ElasticNet | +0.1487 | 0.3053 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| Accuracy_Max_v | Ridge | +0.0707 | 0.6980 | 0.0050 ** | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| Accuracy_Mid_a | ElasticNet | +0.0722 | 0.3344 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| Accuracy_Mid_v | ElasticNet | +0.0306 | 0.8056 | 0.0100 * | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| Neutral_a | Ridge | +0.1541 | 0.2779 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| Neutral_v | ElasticNet | +0.0262 | 0.6865 | 0.0050 ** | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| Speed_Max_a | Ridge | +0.1087 | 0.3343 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| Speed_Max_v | ElasticNet | +0.0291 | 0.8163 | 0.0149 * | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| Speed_Mid_a | ElasticNet | +0.0624 | 0.3053 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| Speed_Mid_v | Ridge | +0.0555 | 0.7760 | 0.0100 * | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
| a | Ridge | +0.1789 | 0.2523 | 0.0050 ** | right_temporal_abs_gamma, central_abs_gamma, right_temporal_rel_gamma, central_rel_gamma, right_temporal_abs_beta, central_abs_beta, prefrontal_abs_beta, prefrontal_abs_gamma, central_abs_delta, posterior_rel_theta |
| v | Lasso | +0.0846 | 0.6018 | 0.0050 ** | right_temporal_abs_gamma, right_temporal_theta_beta_ratio, left_temporal_theta_beta_ratio, right_temporal_rel_gamma, right_temporal_spectral_entropy, right_temporal_rel_beta, central_theta_beta_ratio, central_rel_beta, prefrontal_theta_beta_ratio, central_abs_gamma |
