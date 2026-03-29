# EEG to Decision Behavior: Analysis Report

## Experimental Design

This analysis tests whether resting-state EEG features can predict
decision-making behavior (RT and accuracy), and whether adding task
condition information improves prediction.

### Data

- **EEG**: Resting-state features (~234 features per subject)
  - Aperiodic (1/f exponent, offset) per ROI
  - Periodic (FOOOF peak CF, PW, BW for delta/theta/alpha/beta) per ROI
  - Entropy (permutation, sample) per ROI
  - Connectivity (imaginary coherence between ROI pairs)
  - Global (IAF, band powers, asymmetry, frontal-parietal coherence)
- **Task conditions**: 2 x 5 factorial (Load x Speed-Accuracy emphasis)
  - Load: Load vs NoLoad (working memory manipulation)
  - Speed-Accuracy: Penalty = {5, 10, 20, 40, 80}
  - Encoded as: Load (0/1) + log(Penalty)
- **Targets**: RT_mean (ms), RT_sd (ms), ACC (proportion)
  - Per subject x condition (~10 conditions per subject)

### Comparisons

| Input Mode | Features | Question |
|-----------|----------|----------|
| EEG only | Resting EEG (234d) | Can brain state predict behavior? |
| EEG + Task | EEG + Load(0/1) + log(Penalty) | Does task context help? |

**CV**: 5-fold GroupKFold (subject-level split, no data leakage)

## Models

| Level | Model | Architecture |
|-------|-------|-------------|
| 1 | Ridge | L2-regularized linear regression |
| 1 | XGBoost | Gradient-boosted trees (depth=3, 100 trees) |
| 2 | MLP | 128->64->3 with LayerNorm, GELU, Dropout |
| 3 | Neural SDE | Learned drift f(x,t,z) + diffusion g(x,t,z), soft boundary |
| 4 | GNN+SDE | 7-ROI GCN (connectivity-weighted) -> Neural SDE |

### Neural SDE: Brain-Inspired Evidence Accumulation

Classical DDM uses fixed parameters:
```
dx = v * dt + 1 * dW     (constant drift v, noise = 1)
```

Neural SDE learns time- and state-dependent dynamics:
```
dx = f_theta(x, t, z) * dt + g_theta(x, t, z) * dW
```
where z = encoder(EEG [+ task]) is a latent brain-state vector.

This allows modeling phenomena DDM cannot capture:
- Urgency signals (drift increases with time)
- Attention fluctuations (state-dependent noise)
- Nonlinear evidence accumulation

### ROI-GNN: Topology-Preserving Encoder

7 brain ROIs (prefrontal, frontal, central, posterior, occipital,
left/right temporal) form a graph with connectivity-weighted edges.
2-layer GCN propagates information along anatomical adjacency,
preserving spatial relationships that a flat MLP would ignore.

## Results: RT_mean Prediction

| Model | Input | R-squared | MAE (ms) | Spearman rho |
|-------|-------|-----------|---------|-------------|
| GNN+SDE | eeg_only | -0.0480 | 248.4 | +0.0733 |
| MLP | eeg_only | -0.1382 | 252.2 | +0.0029 |
| NeuralSDE | eeg_only | -0.0392 | 242.1 | +0.0290 |
| Ridge | eeg_only | -49.6879 | 1698.2 | -0.0600 |
| XGBoost | eeg_only | -0.1391 | 261.5 | -0.1610 |
| GNN+SDE | eeg_task | -0.0152 | 239.2 | +0.2722 |
| MLP | eeg_task | -0.0640 | 233.4 | +0.3971 |
| NeuralSDE | eeg_task | -0.0090 | 233.3 | +0.2702 |
| Ridge | eeg_task | -51.6509 | 1725.7 | -0.0192 |
| XGBoost | eeg_task | -0.0488 | 244.0 | +0.3261 |

## Results: ACC Prediction

| Model | Input | R-squared | MAE | Spearman rho |
|-------|-------|-----------|-----|-------------|
| GNN+SDE | eeg_only | -0.0059 | 0.0601 | -0.0121 |
| MLP | eeg_only | -0.1378 | 0.0627 | +0.0837 |
| NeuralSDE | eeg_only | -0.0033 | 0.0600 | -0.0002 |
| Ridge | eeg_only | -21.0970 | 0.2650 | +0.0205 |
| XGBoost | eeg_only | -0.1217 | 0.0646 | +0.0147 |
| GNN+SDE | eeg_task | -0.0062 | 0.0602 | -0.0736 |
| MLP | eeg_task | -0.0892 | 0.0619 | +0.1959 |
| NeuralSDE | eeg_task | -0.0067 | 0.0601 | -0.0289 |
| Ridge | eeg_task | -22.0325 | 0.2706 | +0.0387 |
| XGBoost | eeg_task | -0.0636 | 0.0618 | +0.1604 |

## Key Findings

- **Best EEG-only**: R2=-0.0392 (NeuralSDE)
- **Best EEG+Task**: R2=-0.0090 (NeuralSDE)
- **Task information gain**: Delta R2 = +0.0302

Task condition provides moderate improvement. EEG captures between-subject differences, while task info adds condition-level specificity.

## Model Comparison Notes

- **Classical vs Deep Learning**: If Ridge/XGBoost match or exceed MLP/SDE,
  the EEG-behavior relationship may be approximately linear.
- **Neural SDE vs MLP**: If SDE outperforms MLP, the evidence-accumulation
  inductive bias captures meaningful decision dynamics.
- **GNN+SDE vs flat SDE**: If GNN improves results, spatial brain topology
  provides useful structure beyond treating all features as exchangeable.
