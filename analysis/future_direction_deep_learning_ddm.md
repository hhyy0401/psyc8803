# Predicting Decision-Making Behavior from Resting-State EEG: Analysis Design

## Last Updated: 2026-03-29

---

## 1. Motivation

### 1.1 Problem

Current approach:
```
Resting EEG  -->  Feature Extraction  -->  231 EEG features
Behavioral RT  -->  DDM Fitting  -->  (a, v, t0)

EEG features <-> DDM params  -->  Correlation / Regression
Result: CV R^2 < 0.05, weak individual-level prediction
```

**Limitations:**
1. **DDM bottleneck**: Compressing the full RT distribution into 3 parameters (a, v, t0) loses information
2. **Two-step error propagation**: Uncertainty in DDM fitting is not accounted for in the EEG-DDM regression
3. **Low predictive power**: Resting EEG explains <5% of DDM parameter variance

### 1.2 Key Questions

1. Can EEG + task condition predict RT directly (bypassing DDM)?
2. Does a brain-inspired architecture (Neural SDE) outperform black-box models?
3. Does preserving brain topology (ROI-GNN) add predictive value?

---

## 2. Data

| File | Content | Scale |
|------|---------|-------|
| `resting_subject_features.csv` | EEG features (231d) | 202 subjects |
| `DDM_Scores.csv` | DDM params + behavioral summary per condition | 336 subjects |
| `Behavioral_Scores_trials.csv` | Trial-level RT, ACC | 194,978 trials |
| **Intersection** | | **189 subjects x 10 conditions = 1,890 rows** |

### 2.1 Experimental Design: 2 x 5 Factorial

**Task**: Lexical Decision (word vs non-word discrimination)

- **Load** (2 levels): Load (working memory concurrent task) vs NoLoad
- **Speed-Accuracy** (5 levels): Manipulated via penalty parameter

| Condition | Penalty | Intended Effect |
|-----------|---------|----------------|
| Speed_Max | 5 | Maximize speed: RT decreases, ACC decreases, boundary (a) decreases |
| Speed_Mid | 10 | Moderate speed emphasis |
| Neutral | 20 | Balanced speed-accuracy tradeoff |
| Accuracy_Mid | 40 | Moderate accuracy emphasis |
| Accuracy_Max | 80 | Maximize accuracy: RT increases, ACC increases, boundary (a) increases |

### 2.2 EEG Features (234 total)

Extracted from eyes-closed resting-state EEG (64-ch EGI, ~10 min):

| Category | Features | Description |
|----------|----------|-------------|
| Aperiodic | exponent, offset per ROI | 1/f slope (E/I balance proxy) |
| Periodic | peak CF, PW, BW per band per ROI | FOOOF-extracted oscillatory peaks |
| Band power | alpha, beta, theta, delta | ROI-averaged spectral power |
| Entropy | permutation, sample per ROI | Temporal complexity |
| Connectivity | wPLI between all ROI pairs x 4 bands | Functional coupling (84 features) |
| Global | IAF, asymmetry, coherence | Subject-level summary measures |

**7 ROIs**: Prefrontal, Frontal, Central, Posterior, Occipital, Left Temporal, Right Temporal

### 2.3 Prediction Targets

Per subject x condition:
- **RT_mean** (ms): Mean reaction time
- **RT_sd** (ms): RT variability
- **ACC** (proportion): Accuracy

### 2.4 Task Encoding

Instead of one-hot (10d), we use structured encoding (2d) that preserves ordinal relationships:

```
task_load       = 0 (NoLoad) or 1 (Load)
task_log_penalty = log(Penalty)  -->  {1.6, 2.3, 3.0, 3.7, 4.4}
```

This captures that Speed_Max (penalty=5) and Speed_Mid (penalty=10) are closer to each other than to Accuracy_Max (penalty=80).

---

## 3. Analysis Design

### 3.1 Comparison: EEG Only vs EEG + Task

| Input Mode | Features | Question |
|-----------|----------|----------|
| **EEG only** | 231 EEG features | Can resting brain state predict behavior? |
| **EEG + Task** | 231 EEG + 2 task features | Does task context improve prediction? |

- **EEG only**: Same input for all 10 conditions of a subject. Can only capture between-subject differences.
- **EEG + Task**: Input varies by condition. Can capture condition-specific individual differences.
- **Delta(EEG+Task - EEG only)** = added value of task information

### 3.2 Cross-Validation

5-fold **GroupKFold** with subject-level splitting:
- All 10 conditions of a subject go to train OR test (never split)
- Prevents data leakage from within-subject correlation

### 3.3 Models (4 Levels of Complexity)

#### Level 1: Classical ML

| Model | Description |
|-------|------------|
| **Ridge** | L2-regularized linear regression (RidgeCV, alphas=10^{-3}..10^3) |
| **XGBoost** | Gradient-boosted trees (100 trees, depth=3) |

Separate model per target (RT_mean, RT_sd, ACC).

#### Level 2: MLP (Deep Learning Baseline)

```
Input (n_features) --> Linear(128) --> LayerNorm --> GELU --> Dropout(0.3)
                   --> Linear(64)  --> LayerNorm --> GELU --> Dropout(0.2)
                   --> Linear(3)   --> [RT_mean, RT_sd, ACC]
```

Multi-target prediction. Trained with AdamW, early stopping on validation loss.

#### Level 3: Neural SDE (Brain-Inspired Evidence Accumulation)

**Motivation**: The Drift Diffusion Model (DDM) describes decision-making as evidence accumulation:

```
Classical DDM:  dx = v * dt + 1 * dW     (constant drift v, noise = 1)
```

But real decisions involve time-varying urgency, attention fluctuations, and nonlinear dynamics. The Neural SDE generalizes DDM:

```
Neural SDE:     dx = f_theta(x, t, z) * dt + g_theta(x, t, z) * dW
```

where:
- `x` = accumulated evidence (scalar)
- `t` = time within trial
- `z` = latent brain state (encoded from EEG + task)
- `f_theta` = learned drift function (MLP)
- `g_theta` = learned diffusion function (MLP)

**Architecture**:

```
EEG [+ task]  -->  Encoder (128 -> 64, LayerNorm + GELU)  -->  latent z
                                                                  |
                                              +-------------------+-------------------+
                                              |                   |                   |
                                        drift_net(z,x,t)   diff_net(z,x,t)    boundary_head(z)
                                              |                   |                   |
                                              v                   v                   v
                                         f(x,t,z)           g(x,t,z)         boundary, ndt
                                              |                   |
                                              +----->  SDE simulation  <-----+
                                                    (150 steps, dt=0.01)
                                                    (32 parallel trials)
                                                           |
                                                    Soft boundary crossing
                                                           |
                                                   RT_mean, RT_sd, ACC
```

**Soft boundary**: For differentiable training, we approximate hard boundary crossing with a sigmoid hazard function. At each timestep, the probability of crossing is:

```
hazard(t) = sigmoid(temperature * (|x(t)| - boundary/2))
```

Expected first-passage time is computed as the weighted sum over all timesteps.

**Key differences from DDM**:

| Property | DDM | Neural SDE |
|----------|-----|-----------|
| Drift | Constant v | f(x, t, z) -- varies with time and state |
| Noise | Fixed at 1 | g(x, t, z) -- learned, state-dependent |
| Urgency signals | Cannot model | Natural (drift increases with t) |
| Collapsing boundary | Requires separate implementation | Can emerge from learning |
| Input conditioning | Fit separately per condition | EEG + task directly parameterize dynamics |
| Interpretability | High (3 parameters) | Lower (neural network weights) |

#### Level 4: ROI-GNN + Neural SDE (Topology-Preserving)

**Motivation**: A flat MLP treats all 231 EEG features as exchangeable. But brain features have spatial structure -- frontal theta and frontal alpha come from the same region; frontal-central connectivity reflects specific anatomical pathways.

**Graph construction**:

```
7 ROI nodes:
    prefrontal -- frontal -- central -- posterior -- occipital
                    |           |           |
               left_temporal   |    right_temporal
                    |           |           |
                    +-----------+-----------+

Edge weights = mean |wPLI| across delta/theta/alpha/beta bands
               (from resting-state functional connectivity)
```

- **Node features** (20 per ROI): aperiodic params, periodic peaks, entropy, band power
- **Edge weights**: Functional connectivity (wPLI) -- stronger coupling = higher weight
- **Adjacency**: Symmetric normalized: D^{-1/2} A D^{-1/2}

**Architecture**:

```
ROI node features (7 x 20)  -->  GCN layer 1 (20 -> 64, LayerNorm + GELU)
                             -->  GCN layer 2 (64 -> 64, LayerNorm + GELU)
                             -->  Global mean pool  -->  graph_embed (64d)
                                                              |
                              [graph_embed | global_feat (111d) | task (2d)]
                                                              |
                                                    Fusion --> latent z
                                                              |
                                                    Neural SDE (same as Level 3)
```

GCN propagation: `H^{l+1} = GELU(LN(A * H^l * W^l))`

This preserves spatial relationships: information flows along anatomical pathways weighted by functional connectivity strength.

---

## 4. Results

### 4.1 RT_mean Prediction (Spearman rho)

| Model | EEG only | EEG + Task | Delta |
|-------|----------|-----------|-------|
| Ridge | -0.06 | -0.02 | -- (unstable due to multicollinearity) |
| XGBoost | -0.16 | **+0.33** | **+0.49** |
| MLP | +0.00 | **+0.40** | **+0.40** |
| Neural SDE | +0.03 | +0.27 | +0.24 |
| GNN+SDE | +0.07 | +0.27 | +0.20 |

### 4.2 RT_mean Prediction (R-squared)

| Model | EEG only | EEG + Task |
|-------|----------|-----------|
| Ridge | -49.69 | -51.65 |
| XGBoost | -0.14 | -0.05 |
| MLP | -0.14 | -0.06 |
| Neural SDE | -0.04 | **-0.01** |
| GNN+SDE | -0.05 | -0.02 |

Note: R-squared values are all negative, indicating predictions are worse than the
global mean in absolute scale. However, Spearman rho shows the models capture
the correct **ranking** of RT values. This is expected given that resting-state EEG
has inherently limited predictive power for behavioral outcomes (prior best R^2 = 0.044).

### 4.3 RT_sd and ACC Prediction (EEG + Task, Spearman rho)

| Model | RT_sd rho | ACC rho |
|-------|----------|---------|
| XGBoost | **+0.32** | +0.16 |
| MLP | +0.31 | **+0.20** |
| Neural SDE | +0.19 | -0.03 |
| GNN+SDE | +0.21 | -0.07 |

### 4.4 Key Findings

**1. Task information is decisive.**
EEG-only models produce rho near 0 for all architectures. Adding just 2 task features
(Load + log Penalty) raises the best rho from ~0 to 0.40. This is expected: EEG-only
input is identical across all 10 conditions per subject, so the model cannot
distinguish condition-specific RT variation.

**2. MLP achieves the best performance (rho = 0.40).**
A simple 2-layer MLP outperforms all other models, including the brain-inspired
Neural SDE (rho = 0.27) and GNN+SDE (rho = 0.27).

**3. Neural SDE does not outperform MLP.**
The evidence-accumulation inductive bias does not help -- and slightly hurts --
prediction accuracy. Possible reasons:
- The soft-boundary SDE simulation introduces optimization difficulty
  (150-step sequential computation with stochastic noise)
- With only ~1,890 samples, the simpler MLP generalizes better
- The SDE's interpretable structure (drift, boundary, ndt) may be overly constrained
  for this prediction task

**4. GNN does not improve over flat SDE.**
ROI-level spatial topology (7-node graph with connectivity-weighted edges) does not
add predictive value beyond what a flat feature vector provides. This suggests that
for condition-level RT prediction, the spatial organization of EEG features is less
important than the features themselves.

**5. Ridge regression fails catastrophically (R^2 = -50).**
With 231 highly correlated EEG features, RidgeCV selects insufficient regularization,
leading to severe overfitting on training subjects. XGBoost's tree-based feature
selection naturally handles this multicollinearity.

### 4.5 Interpretation

The core finding is that **resting-state EEG alone cannot predict condition-specific
decision-making behavior** (EEG-only rho near 0). When task condition is provided,
a simple MLP achieves moderate ranking accuracy (rho = 0.40), suggesting the model
learns subject-specific sensitivity to task demands from EEG features.

The negative R-squared values across all models indicate that resting-state EEG,
even combined with task information, cannot predict the absolute scale of RT
reliably. This is consistent with the broader literature showing weak
individual-level EEG-behavior associations (typical R^2 < 0.10).

The brain-inspired architectures (Neural SDE, GNN) do not outperform a vanilla MLP.
This suggests that for this dataset and sample size, model complexity does not
translate to better prediction -- the bottleneck is the signal strength in resting
EEG, not the model architecture.

---

## 5. Future Directions

1. **Larger sample size**: N=189 may be insufficient for complex architectures (SDE, GNN) to show their advantage over simpler models.
2. **Task-evoked EEG**: Pre-stimulus or during-task EEG features could provide trial-level predictors that resting-state EEG cannot.
3. **Pre-training**: Pre-train the SDE encoder on a larger EEG dataset, then fine-tune on decision-making prediction.
4. **Hybrid models**: Use MLP for prediction but regularize with DDM-inspired losses (e.g., predicted RT distribution should be plausible under DDM constraints).

---

## 6. References

- **wPLI as GNN edge weights**: Klepl et al. (2022). EEG-Based Graph Neural Network Classification of Alzheimer's Disease. *IEEE TNSRE*, 30, 2651-2660.
- **ROI-level brain graphs**: Li et al. (2021). BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis. *Medical Image Analysis*, 74, 102233.
- **FC-based GCN for cognition**: Panwar et al. (2024). EEG-CogNet: A deep learning framework for cognitive state assessment. *Biomedical Signal Processing and Control*.
- **DDM and EEG**: Euler et al. (2024); Pathania et al. (2022) -- Aperiodic exponent predicts drift rate.
- **Neural SDEs**: Chen et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
