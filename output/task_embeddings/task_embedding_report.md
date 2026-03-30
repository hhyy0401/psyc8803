# Task Embedding Experiment Report

Generated: 2026-03-30 14:46

## Overview

Compares five task encoding strategies for EEG + task → RT prediction:

| Tier | Label | Dim | Description |
|------|-------|-----|-------------|
| 0 | Baseline | 2 | [load, log(penalty)] — current approach |
| 1 | SDT/EV | 5 | [load, log(R/P), τ_acc, strategy_ord, load×log(R/P)] |
| 2 | DDM | 8 | Tier1 + [r_net, Δa*, rt_direction] |
| 3 | SBERT | 8 | Sentence-BERT instruction text → PCA-8 |
| 4 | Learned | 8 | nn.Embedding(10,8) warm-started from Tier1 |

Models tested: **XGBoost** (Tiers 0–3), **MLP** (Tiers 0–4)

Cross-validation: 5-fold GroupKFold (subject-level splits)

## Results: Spearman ρ (RT mean, RT sd, ACC)

| Embedding | RT_mean ρ | RT_sd ρ | ACC ρ |
|-----------|-----------|---------|-------|
| XGB-T0 | +0.3333 | +0.3114 | +0.1251 |
| XGB-T1 | +0.3383 | +0.3279 | +0.1403 |
| XGB-T2 | +0.3383 | +0.3279 | +0.1403 |
| XGB-T3 | +0.3460 | +0.3190 | +0.1230 |
| MLP-T0 | +0.3711 | +0.3253 | +0.1923 |
| MLP-T1 | +0.4093 | +0.3604 | +0.2319 |
| MLP-T2 | +0.3867 | +0.3271 | +0.2112 |
| MLP-T3 | +0.3711 | +0.3305 | +0.1919 |
| MLP-T4 | +0.3752 | +0.3324 | +0.2000 |

## Δρ vs Tier-0 Baseline (RT_mean)

| Embedding | Δρ (RT_mean) | Δρ (RT_sd) | Δρ (ACC) |
|-----------|-------------|------------|----------|
| XGB-T1 | +0.0050 | +0.0165 | +0.0151 |
| XGB-T2 | +0.0050 | +0.0165 | +0.0151 |
| XGB-T3 | +0.0127 | +0.0076 | -0.0021 |
| MLP-T1 | +0.0382 | +0.0351 | +0.0395 |
| MLP-T2 | +0.0156 | +0.0018 | +0.0189 |
| MLP-T3 | +0.0000 | +0.0052 | -0.0004 |
| MLP-T4 | +0.0041 | +0.0071 | +0.0077 |

## SBERT PCA: Explained Variance

| PC | Explained Variance | Cumulative |
|----|-------------------|------------|
| PC1 | 45.1% | 45.1% |
| PC2 | 41.8% | 86.8% |
| PC3 | 7.0% | 93.9% |
| PC4 | 3.8% | 97.7% |
| PC5 | 1.3% | 99.0% |
| PC6 | 0.5% | 99.5% |
| PC7 | 0.2% | 99.8% |
| PC8 | 0.2% | 99.9% |

## Figures

| Figure | Description |
|--------|-------------|
| fig1_task_feature_values.png | Tier-1 and Tier-2 feature values by condition |
| fig2_sbert_pca.png | SBERT instruction embeddings (PCA-2) |
| fig3_rho_comparison.png | Spearman ρ bar chart for all tiers/models |
| fig4_delta_rho.png | Δρ gain over Tier-0 baseline |
| fig5_heatmap.png | Heatmap: models × targets |
| fig6_scatter_rtmean.png | Actual vs predicted RT_mean (best tier per model) |
| fig7_scatter_acc.png | Actual vs predicted ACC (best tier per model) |
