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

---

## 7. Task Embedding Plan

### 7.1 Motivation: Limitations of the Current 2D Encoding

The current task encoding is:
```
task_load        = 0 (NoLoad) or 1 (Load)
task_log_penalty = log(|penalty|)  -->  {1.61, 2.30, 3.00, 3.69, 4.38}
```

From the TaskDescription PDF, the full experimental structure is a **2x5 factorial**:

| Condition | Load | Penalty | Reward | Instruction Framing |
|-----------|------|---------|--------|---------------------|
| Speed_Max | 0/1 | 5 | 20 | "answer quickly, errors not penalized very much" |
| Speed_Mid | 0/1 | 10 | 20 | "try to answer relatively quickly" |
| Neutral | 0/1 | 20 | 20 | "try to earn as many points as possible" |
| Accuracy_Mid | 0/1 | 40 | 20 | "try to answer relatively carefully" |
| Accuracy_Max | 0/1 | 80 | 20 | "answer carefully, errors will be heavily penalized" |

The 2D encoding has three blind spots:

1. **No normative grounding.** `log_penalty` captures *ordinal ranking* but not
   what the task actually demands from the subject in terms of optimal behavior.
   A subject who understands the task knows they need to be ≥80% accurate in
   Accuracy_Max to avoid losing points. That threshold is not encoded.

2. **Load is underspecified.** The Load manipulation is specifically an auditory
   phonological n-back (hear a letter during each feedback display, recall the
   last two on probe trials ~every 8 trials). This is a phonological WM load,
   not a generic binary flag. Any interaction between Load and speed-accuracy
   strategy is invisible in a pure 0/1 encoding.

3. **No instruction semantics.** Subjects receive explicitly different textual
   instructions per block. The framing ("answer quickly" vs. "answer carefully")
   directly primes strategy. This semantic content is entirely discarded.

---

### 7.2 Tier 1 — Normative EV / SDT-Grounded Encoding (2D → 5D, drop-in)

**Core idea**: Augment the current features with quantities derived from the
expected-value framework implied by the reward structure. These are
deterministic functions of Penalty and Reward (both provided in `DDM_Scores.csv`).

#### New features

**Break-even accuracy threshold** (SDT-motivated):
$$\tau_{acc} = \frac{|P|}{R + |P|}$$

The minimum accuracy a subject must achieve to have non-negative expected score
per trial. Under expected value maximization, this is the threshold that divides
"it is worth responding at all" from "guessing randomly".

| Condition | Penalty | τ_acc |
|-----------|---------|-------|
| Speed_Max | 5 | 5/25 = 0.20 |
| Speed_Mid | 10 | 10/30 = 0.33 |
| Neutral | 20 | 20/40 = 0.50 |
| Accuracy_Mid | 40 | 40/60 = 0.67 |
| Accuracy_Max | 80 | 80/100 = 0.80 |

This nonlinear transform of penalty has different slope structure from
`log_penalty` — it compresses the speed end and stretches the accuracy end
in a sigmoidal curve, which better matches the increasing difficulty of achieving
higher accuracy thresholds.

**Log reward-to-penalty ratio** (DDM-motivated, symmetric):
$$\text{log\_rp} = \log\!\left(\frac{R}{|P|}\right)$$

Centered at 0 for Neutral (R=P=20), positive for speed conditions, negative for
accuracy conditions. This makes the neutrality of the Neutral condition algebraically
explicit, which may help models learn symmetric treatment of speed vs. accuracy.

| Condition | Penalty | log(R/P) |
|-----------|---------|---------|
| Speed_Max | 5 | log(4) = +1.39 |
| Speed_Mid | 10 | log(2) = +0.69 |
| Neutral | 20 | log(1) = 0.00 |
| Accuracy_Mid | 40 | log(0.5) = -0.69 |
| Accuracy_Max | 80 | log(0.25) = -1.39 |

Note: `log(R/P) = log(20) - log_penalty`, so this is a linear transform of the
current feature. Its value is as a **centred and interpretable** replacement for
`task_log_penalty` (rather than an additional feature).

**Strategy ordinal** (explicit category):
$$s = \{0,\ 0.25,\ 0.5,\ 0.75,\ 1.0\} \text{ for } \{\text{SpMax, SpMid, Neut, AccMid, AccMax}\}$$

Adds a smooth ordinal axis that is linearly spaced (unlike log_penalty or τ_acc),
giving models an alternative graduation to learn from.

**Load × log(R/P) interaction**:
$$\text{load\_rp} = \text{task\_load} \times \log(R/|P|)$$

The phonological WM load imposed in the Load condition may interact multiplicatively
with speed-accuracy demands. A subject under WM load responding in Speed_Max
(fastest, easiest strategy for the lexical task) faces a qualitatively different
cognitive trade-off from a subject under WM load in Accuracy_Max (requiring
both phonological WM and response caution simultaneously).

#### Full Tier 1 feature vector (5D)

```python
import numpy as np

def tier1_task_features(load, penalty, reward=20):
    """
    load    : 0 or 1
    penalty : absolute value (5, 10, 20, 40, 80)
    reward  : 20 (constant in this study)
    Returns : np.array of shape (5,)
    """
    log_rp        = np.log(reward / penalty)               # centred on 0 at Neutral
    tau_acc       = penalty / (reward + penalty)           # break-even accuracy
    strategy_ord  = {5: 0.00, 10: 0.25, 20: 0.50,         # ordinal strategy level
                     40: 0.75, 80: 1.00}[penalty]
    load_rp_inter = float(load) * log_rp                   # WM load × SAT interaction
    return np.array([float(load), log_rp, tau_acc, strategy_ord, load_rp_inter])
```

Code change required in `analysis_rt.py` `load_data()`:
```python
# Replace current 2-line task encoding with:
merged["task_load"]        = (merged["Load_Condition"] == "Load").astype(float)
merged["task_log_rp"]      = np.log(merged["Reward"].astype(float) /
                                    merged["Penalty"].astype(float))
merged["task_tau_acc"]     = (merged["Penalty"].astype(float) /
                               (merged["Reward"].astype(float) +
                                merged["Penalty"].astype(float)))
_ord_map = {5: 0.00, 10: 0.25, 20: 0.50, 40: 0.75, 80: 1.00}
merged["task_strategy_ord"] = merged["Penalty"].map(_ord_map)
merged["task_load_rp"]     = merged["task_load"] * merged["task_log_rp"]

TASK_COLS = ["task_load", "task_log_rp", "task_tau_acc",
             "task_strategy_ord", "task_load_rp"]
```

---

### 7.3 Tier 2 — DDM-Theoretic Optimal Parameter Encoding (5D → 8D)

**Core idea**: Encode what an optimal DDM agent *would* do under each condition's
reward structure, without knowing the subject's actual drift rate `v`. We compute
relative normative targets: how much should boundary, drift threshold, and RT
shift relative to the Neutral condition?

#### Theory

Under the DDM, the expected pointwise reward rate is:

$$ER(a, v) = \frac{R \cdot P_c(a,v) - |P| \cdot P_e(a,v)}{E[DT](a,v) + t_0 + \delta}$$

where $P_c = \frac{1}{1+e^{-2av}}$, $E[DT] = \frac{a}{v}\tanh(av)$, and $\delta$
is the intertrial interval. The optimal boundary $a^*(v, R, P)$ that maximises $ER$
scales with the penalty ratio. For fixed $v$, higher $|P|/R$ → larger $a^*$.

Since $v$ is unknown a priori, we encode the *relative shift* in optimal boundary
from Neutral $(R = |P|)$. A first-order approximation gives:

$$\Delta a^*_{\rm rel} = \sqrt{\frac{|P|/R + 1}{2}} - 1$$

This is positive for accuracy conditions (penalty > reward → larger boundary) and
negative for speed conditions, with value 0 at Neutral. It is a smooth, monotonic
function of the reward-to-penalty ratio.

#### Additional DDM-motivated features

**Expected score per correct trial** (normalised):
$$r_{net} = \frac{R - |P|}{R + |P|} = \frac{20 - |P|}{20 + |P|}$$

Ranges from +0.6 (Speed_Max: net +15 per 25) to −0.6 (Accuracy_Max: net −60 per 100).
This is the normalised net expected value under 50% accuracy, encoding how far the
condition is from break-even.

**Relative optimal boundary shift** (above):
$$\Delta a^* = \sqrt{(|P|/R + 1)/2} - 1$$

| Condition | Penalty | Δa* |
|-----------|---------|-----|
| Speed_Max | 5 | −0.21 |
| Speed_Mid | 10 | −0.13 |
| Neutral | 20 | 0.00 |
| Accuracy_Mid | 40 | +0.22 |
| Accuracy_Max | 80 | +0.42 |

**Predicted RT direction** (sign of a* shift, useful for regression targets):
$$\text{rt\_direction} = \text{sign}(\Delta a^*) \in \{-1, 0, +1\}$$

#### Full Tier 2 additions (3 new features appended to Tier 1)

```python
def tier2_task_features(load, penalty, reward=20):
    t1 = tier1_task_features(load, penalty, reward)
    rp_ratio     = penalty / reward
    r_net        = (reward - penalty) / (reward + penalty)
    delta_a_star = np.sqrt((rp_ratio + 1) / 2) - 1
    rt_direction = np.sign(delta_a_star)
    return np.concatenate([t1, [r_net, delta_a_star, rt_direction]])
```

---

### 7.4 Tier 3 — Sentence Embedding of Instruction Text (8D projected)

**Core idea**: Each block begins with the instruction text that participants
actually read. The semantic framing — "answer quickly", "try to answer carefully",
"errors will be heavily penalized" — is the direct manipulation. Encoding this
text captures the *communicated strategy* rather than only the underlying penalty
structure.

#### Instruction texts (from TaskDescription.pdf)

```python
INSTRUCTION_TEXTS = {
    ("Speed_Max",    "NoLoad"): "To maximize your score, answer quickly, "
                                "because errors will not be penalized very much.",
    ("Speed_Max",    "Load"):   "To maximize your score, answer quickly, "
                                "because errors will not be penalized very much. "
                                "Also, there will be the additional task of "
                                "memorizing letters presented during feedback.",
    ("Speed_Mid",    "NoLoad"): "To maximize your score, try to answer relatively "
                                "quickly, because errors will not be penalized very much.",
    ("Speed_Mid",    "Load"):   "To maximize your score, try to answer relatively "
                                "quickly, because errors will not be penalized very much. "
                                "Also, there will be the additional task of "
                                "memorizing letters presented during feedback.",
    ("Neutral",      "NoLoad"): "Try to earn as many points as possible to "
                                "maximize your score.",
    ("Neutral",      "Load"):   "Try to earn as many points as possible to "
                                "maximize your score. Also, there will be the "
                                "additional task of memorizing letters presented "
                                "during feedback.",
    ("Accuracy_Mid", "NoLoad"): "To maximize your score, try to answer relatively "
                                "carefully, because errors will be heavily penalized.",
    ("Accuracy_Mid", "Load"):   "To maximize your score, try to answer relatively "
                                "carefully, because errors will be heavily penalized. "
                                "Also, there will be the additional task of "
                                "memorizing letters presented during feedback.",
    ("Accuracy_Max", "NoLoad"): "To maximize your score, answer carefully, "
                                "because errors will be heavily penalized.",
    ("Accuracy_Max", "Load"):   "To maximize your score, answer carefully, "
                                "because errors will be heavily penalized. "
                                "Also, there will be the additional task of "
                                "memorizing letters presented during feedback.",
}
```

#### Encoding pipeline

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

def build_sbert_task_embeddings(n_components=8):
    model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-d, fast, MIT licence

    conditions = list(INSTRUCTION_TEXTS.keys())
    texts      = [INSTRUCTION_TEXTS[c] for c in conditions]
    embeds     = model.encode(texts, normalize_embeddings=True)  # (10, 384)

    # Project down with PCA; fix number of components (8 retains ~90%+ variance
    # across 10 conditions, verified empirically)
    pca    = PCA(n_components=n_components, random_state=42)
    embeds_reduced = pca.fit_transform(embeds)         # (10, 8)

    cond_to_embed = {c: embeds_reduced[i] for i, c in enumerate(conditions)}
    return cond_to_embed, pca

def apply_sbert_features(merged, cond_to_embed):
    """Map each row's (Speed_Condition, Load_Condition) to its 8-d embedding."""
    def row_embed(row):
        key = (row["Speed_Condition"], row["Load_Condition"])
        return cond_to_embed[key]
    embed_matrix = np.stack(merged.apply(row_embed, axis=1).values)
    cols = [f"sbert_{i}" for i in range(embed_matrix.shape[1])]
    return pd.DataFrame(embed_matrix, columns=cols, index=merged.index)
```

**Why this is not redundant with Tier 1**: The SBERT embedding captures *linguistic
framing*, not just penalty arithmetic. In particular, it differentiates the
"relatively quickly" phrasing of Speed_Mid from the "answer quickly" of Speed_Max
in a continuous vector space, and it naturally groups Load conditions with their
non-Load counterparts through the shared penalty-related sentence stem.

**Pre-computation note**: Since there are only 10 unique condition texts, the
SBERT forward pass runs once at startup (not per sample), introducing negligible
overhead. The PCA weights are fixed after the first run.

---

### 7.5 Tier 4 — Learned Condition Embeddings (trainable, 8D)

**Core idea**: For neural models (MLP, Neural SDE, GNN+SDE), replace the hand-
engineered task vector with a trainable `nn.Embedding(10, d_task)` lookup. The 10
conditions are indexed deterministically (see mapping below), and the embedding
vectors are jointly optimised with the rest of the model.

#### Condition index mapping

```python
CONDITION_INDEX = {
    ("Speed_Max",    "NoLoad"): 0,
    ("Speed_Max",    "Load"):   1,
    ("Speed_Mid",    "NoLoad"): 2,
    ("Speed_Mid",    "Load"):   3,
    ("Neutral",      "NoLoad"): 4,
    ("Neutral",      "Load"):   5,
    ("Accuracy_Mid", "NoLoad"): 6,
    ("Accuracy_Mid", "Load"):   7,
    ("Accuracy_Max", "NoLoad"): 8,
    ("Accuracy_Max", "Load"):   9,
}
```

#### Module

```python
class LearnedTaskEmbedding(nn.Module):
    def __init__(self, n_conditions=10, d_task=8, init_features=None):
        """
        init_features : optional (10, d_structured) array for warm-start.
                        If provided, a linear projection + LayerNorm maps it
                        to d_task dimensions and initialises the embedding table.
        """
        super().__init__()
        self.emb = nn.Embedding(n_conditions, d_task)
        if init_features is not None:
            proj = nn.Linear(init_features.shape[1], d_task, bias=False)
            with torch.no_grad():
                projected = proj(torch.FloatTensor(init_features))
                self.emb.weight.copy_(projected)
        else:
            nn.init.normal_(self.emb.weight, std=0.01)

    def forward(self, condition_idx):
        return self.emb(condition_idx)   # (B, d_task)
```

**Integration point**: In `MLPModel`, replace the raw task feature concatenation:
```python
# Before (hardcoded 2-D task concatenation in load_data):
x = torch.cat([eeg_feat, task_feat_2d], dim=-1)

# After (learned embedding lookup):
task_emb = self.task_embedding(condition_idx)    # (B, 8)
x = torch.cat([eeg_feat, task_emb], dim=-1)
```

#### Warm-start option (Tier 5 / Hybrid)

Initialise the embedding table from the Tier 1 or Tier 2 structured features (via
the `init_features` argument above). This is the recommended default: it provides
a principled starting point that respects the ordinal penalty structure while
allowing gradients to refine the representation toward task-specific prediction.

---

### 7.6 Comparison Summary and Expected Gains

#### Dimension table

| Tier | Method | Dimension | New information beyond current |
|------|--------|-----------|--------------------------------|
| Current | [load, log_penalty] | 2D | — |
| Tier 1 | EV / SDT features | 5D | Break-even threshold, symmetric RP ratio, ordinal strategy, Load×SAT interaction |
| Tier 2 | + DDM-theoretic targets | 8D | Normative boundary shift, net EV, RT direction |
| Tier 3 | SBERT instruction embed (PCA) | 8D | Instruction semantics, linguistic framing |
| Tier 4 | Learned embedding | 8D | Fully data-driven; no structure imposed |
| Tier 5 | Warm-start learned (Tier 1 init) | 8D | Data-driven refinement from principled init |

#### Hypotheses

**Tier 1 will improve XGBoost most.** Tree-based models can exploit the nonlinear
τ_acc feature directly, without needing a neural network to learn the transformation
from log_penalty. We expect Δrho ≈ +0.02 to +0.05 purely from adding the break-even
accuracy feature.

**Tier 3 (SBERT) will add marginal gain over Tier 1 for MLP.** The instruction
semantic dimensions are nearly linearly predictable from penalty since the instructions
were designed to match the penalty values. However, the Load description adds a
distinct semantic axis (phonological monitoring) not present in the Tier 1 features,
which may help separate Load from NoLoad conditions in latent space.

**Tier 4/5 will only help if the current rho ceiling is not already near the
fundamental EEG-behavior limit.** Given that MLP already achieves rho = 0.40 with
2D task features, the bottleneck is more likely the EEG→behavior signal strength
than the task encoding. Nevertheless, Tier 5 (warm-start) carries no downside
risk — it cannot do worse than Tier 1 in expectation because gradients will simply
leave the pretrained direction undisturbed if the structured features are already
sufficient.

#### Recommended implementation order

1. **Start with Tier 1** (5-minute code change in `load_data()`, no new dependencies).
   Run the full model comparison. If Δrho > 0 for any model, adopt as the new baseline.

2. **Add Tier 2** (pure arithmetic, no new dependencies).
   Merge into the Tier 1 vector (8D total). Re-run comparison.

3. **Add Tier 3** (requires `sentence-transformers` and `scikit-learn` PCA, both
   already in likely env). Treat as a separate ablation column alongside Tier 2
   (not stacked additively, to avoid dimensionality explosion).

4. **Add Tier 4/5** for the MLP and Neural SDE only (not Ridge/XGBoost, which use
   tabular features). Use Tier 1 warm-start as default. Monitor training curve for
   embedding collapse (all 10 condition vectors converging to similar directions).

#### Ablation table template (to be filled after experiments)

| Embedding | XGB rho | MLP rho | SDE rho | GNN+SDE rho |
|-----------|---------|---------|---------|-------------|
| Tier 0 – Current (2D) | +0.333 | +0.371 | +0.27† | +0.27† |
| Tier 1 – SDT/EV (5D) | +0.338 | **+0.409** | — | — |
| Tier 2 – DDM (8D) | +0.338 | +0.387 | — | — |
| Tier 3 – SBERT (8D) | +0.346 | +0.371 | — | — |
| Tier 4 – Learned warm (8D) | — | +0.375 | — | — |

RT_mean Spearman ρ from 5-fold GroupKFold CV (N=189 subjects, 1890 rows).
† SDE / GNN+SDE numbers from prior analysis_rt.py run (not re-run here).
Best result per family highlighted in **bold**.
