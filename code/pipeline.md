# EEG → DDM Prediction Pipeline

## Overview

Predict Drift Diffusion Model (DDM) parameters (**a**, **v**, **t0**) from
resting-state EEG using spectral (FOOOF), connectivity (wPLI + coherence),
entropy, and deep learning approaches.

- **N subjects**: ~193 preprocessed, ~181 overlap with DDM
- **EEG**: 64-channel EGI HydroCel GSN, eyes-closed resting state (~10–12 min)
- **DDM targets**: boundary separation (a), drift rate (v), non-decision time (t0)

---

## 1. Pipeline Overview

```
Step 1: preprocessing.py          (raw .mff → clean epochs .fif)
Step 2: extract_features.py       (clean epochs → features.csv + psd_matrix.npz)
Step 3: regression.py             (features + DDM → prediction results)
```

---

## 2. Step 1 — Preprocessing (`preprocessing.py`)

**Author**: Yoonsang Lee
**Status**: Complete (193 subjects in `raw_clean/`)

```
Raw MFF → read_raw_egi()
       → crop eyes-closed segment (BAD_ACQ_SKIP, discard first 10s)
       → pick EEG channels (E1–E64)
       → resample to 250 Hz
       → notch filter 60 Hz
       → bandpass filter 1.0–40 Hz
       → bad channel detection (variance + 60Hz line noise) → interpolate
       → average reference
       → fixed-length epochs (2 sec, 50% overlap)
       → reject bad epochs (threshold 200 µV)
       → save {subject}_clean-epo.fif
```

**Input**: `raw/*.mff` (203 files, 59 GB)
**Output**: `raw_clean/*_clean-epo.fif` (193 files, ~1.8 GB)

---

## 3. Step 2 — Feature Extraction (`extract_features.py`)

Loads clean epoch files, extracts all EEG features.

### 3.1 Features extracted (~231 total)

| Category | Features | Count |
|----------|----------|-------|
| **Band power** (log mean) | posterior α/β, frontal θ/α, TBR, prefrontal θ, central α, occipital α, global δ/θ/α/β | 12 |
| **IAF** | Peak alpha frequency (posterior, 7–13 Hz) | 1 |
| **Temporal α asymmetry** | R–L log alpha power | 1 |
| **FOOOF per ROI** | 7 ROIs × (aperiodic offset/exponent + δ/θ/α/β peak CF/PW/BW) | 98 |
| **FOOOF asymmetry** | (R–L)/(|R|+|L|) for L/R temporal, 14 FOOOF params | 14 |
| **Specific connectivity** | fronto-posterior + L-R temporal (wPLI+coh, θ+α) | 7 |
| **Full wPLI** | 21 ROI pairs × 4 bands (δ/θ/α/β) | 84 |
| **Entropy** | Sample entropy + permutation entropy × 7 ROIs | 14 |
| **Total** | | **~231** |

### 3.2 FOOOF settings

```python
peak_width_limits = [1, 12]
max_n_peaks = 6
min_peak_height = 0.1
aperiodic_mode = "fixed"
freq_range = [1, 40]
bands: delta(1-4), theta(4-7), alpha(8-12), beta(13-30)
```

### 3.3 Connectivity

- **wPLI** (weighted Phase Lag Index): robust to volume conduction
- **Coherence**: magnitude-squared coherence for specific pairs
- **Full wPLI**: all 21 ROI pairs via ROI-averaged virtual channels, multitaper

### 3.4 Entropy

- **Sample entropy**: temporal regularity (via antropy)
- **Permutation entropy**: complexity of ordinal patterns (normalized)

### 3.5 Output

| File | Format | Description |
|------|--------|-------------|
| `output/features.csv` | CSV | (N × ~232) feature matrix (`subject` + ~231 features) |
| `output/psd_matrix.npz` | NPZ | (N, 7, ~79) ROI-averaged PSD for 1D-CNN |

---

## 4. Step 3 — Regression (`regression.py`)

### 4.1 DDM Targets (24 total)

| Category | Targets | Count |
|----------|---------|-------|
| Grand mean | mean(a, v, t0) across all 10 conditions | 3 |
| Load / NoLoad | mean per load condition × 3 params | 6 |
| Condition-paired | 5 speed conditions × 3 params | 15 |

### 4.2 Models (10)

**ML (7)** — with permutation test (100 iter):

| Model | Tuning | Notes |
|-------|--------|-------|
| Ridge | RidgeCV (50 alphas, built-in) | Fast, no grid search |
| Lasso | LassoCV (L1, built-in) | Sparse selection |
| ElasticNet | ElasticNetCV (L1+L2) | Balanced |
| KernelRidge | GridSearchCV (alpha, kernel, gamma) | Non-linear kernel |
| SVR | GridSearchCV (C, epsilon, rbf) | Support vectors |
| XGBoost | GridSearchCV (depth, lr, reg) | Gradient boosting |
| RandomForest | GridSearchCV (trees, depth, leaf) | Ensemble |

**DL (3)** — no permutation test:

| Model | Architecture | Input |
|-------|-------------|-------|
| MLP | 128→BN→ReLU→Drop(.5)→64→BN→ReLU→Drop(.5)→1 | Top-15 features |
| 1D-CNN | Conv(7→32,k5)→Conv(32→64,k5)→AvgPool→FC(64→32→1) | Raw PSD (7×79) |
| TabNet | n_d=8, n_a=8, n_steps=3 | Top-15 features |

### 4.3 Evaluation

- **Cross-validation**: 10-fold KFold (shuffle, seed=42)
- **Feature selection**: Top-15 by |Pearson r| within each fold (training only)
- **Metrics**: CV R², Spearman ρ, Pearson r, MAE
- **Permutation test**: 100 iterations, parallel (joblib), ML only

### 4.4 Output

| File | Description |
|------|-------------|
| `output/results/regression_results.csv` | All 24 targets × 10 models |
| `output/results/regression_report.md` | Best models, top features, summary |

---

## 5. ROI Definitions

```
HydroCel GSN 64 1.0

Frontal:        E3, E6, E8, E9, E11, E2
Posterior:      E34, E31, E40, E33, E38, E36
Central:        E16, E7, E4, E54, E51, E41, E21
Left Temporal:  E22, E24, E25, E26, E27
Right Temporal: E49, E52, E48, E46, E45
Occipital:      E36, E37, E39, E32, E43
Prefrontal:     E1, E17, E2, E11, E5, E10
```

---

## 6. Execution

```bash
conda activate base

# Step 1: Already done (193 subjects in raw_clean/)
# python code/preprocessing.py

# Step 2: Feature extraction (~15–20 min)
python code/extract_features.py

# Step 3: Regression (~2–3 hours)
pip install pytorch-tabnet   # optional
python code/regression.py
```

### Estimated runtime

| Step | Duration |
|------|----------|
| Feature extraction (FOOOF + wPLI + entropy × 193 subjects, 4 workers) | ~15–20 min |
| Regression main CV (24 targets × 10 models × 10 folds) | ~30 min |
| Permutation test (24 × 7 ML × 100 perm, parallel) | ~60–90 min |
| DL models (24 × 3 DL × 10 folds) | ~20 min |
| **Total** | **~2–2.5 hours** |

---

## 7. Key Design Decisions

1. **Two-stage preprocessing**: `preprocessing.py` handles raw EEG cleaning
   (bad channels, filtering, artifact rejection), `extract_features.py`
   handles feature computation. Clean epochs are cached as .fif files.

2. **FOOOF + traditional band power**: Both parameterized (FOOOF) and
   classical log-power features are extracted for comprehensive comparison.

3. **Multiple connectivity measures**: wPLI (robust to volume conduction)
   for all ROI pairs + coherence for specific hypothesis-driven pairs.

4. **Entropy features**: Sample entropy and permutation entropy capture
   temporal dynamics beyond spectral power.

5. **t0 as target**: Non-decision time is largely unexplored in
   resting-state EEG literature.

6. **1D-CNN on raw PSD**: End-to-end learning from spectral representation
   bypasses hand-crafted features.

---

## 8. Domain Knowledge Alignment

| Hypothesis | Evidence | Status |
|-----------|---------|--------|
| Aperiodic exponent → drift rate (v) | ★★★ (Euler 2024) | To verify with full features |
| IAF → processing speed | ★★ (Grandy 2013) | Now extracted |
| TBR → boundary separation (a) | ★★ (literature) | Prior: predicted v instead |
| Connectivity → DDM | Theoretical | **New: wPLI + coh included** |
| Entropy → DDM | Emerging | **New: sample + perm entropy** |
| Temporal ROI dominance | Unexpected prior finding | To investigate |

---

## 9. Future Directions

1. **HDDM**: If trial-level behavioral data available, fit hierarchical
   Bayesian DDM with EEG covariates (joint model).
2. **Muscle artifact control**: Temporal gamma/beta may reflect EMG.
3. **Cross-frequency coupling**: Theta-gamma PAC.
4. **Task-state EEG**: Event-related features for more direct prediction.
5. **Multi-modal**: EEG + demographics + behavioral measures.
