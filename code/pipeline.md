# EEG Analysis Pipeline

## Overview

Predict decision-making behavior from resting-state EEG using spectral (FOOOF),
connectivity (wPLI + coherence), entropy, and deep learning approaches.

- **N subjects**: ~193 preprocessed, ~189 overlap with DDM + behavioral
- **EEG**: 64-channel EGI HydroCel GSN, eyes-closed resting state (~10–12 min)
- **Targets**: DDM parameters (a, v, t0) and behavioral outcomes (RT, ACC)

---

## 1. Pipeline Overview

```
Step 1: preprocessing.py          (raw .mff → clean epochs .fif)
Step 2: extract_features.py       (clean epochs → features.csv + psd_matrix.npz)
Step 3: analysis_features.py      (EEG features ↔ DDM params: correlation + regression)
Step 4: analysis_rt.py            (EEG [+ task] → RT/ACC prediction with SDE/GNN)
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

## 4. Step 3 — Feature–DDM Analysis (`analysis_features.py`)

### 4.1 DDM Targets (24 total)

| Category | Targets | Count |
|----------|---------|-------|
| Grand mean | mean(a, v, t0) across all 10 conditions | 3 |
| Load / NoLoad | mean per load condition × 3 params | 6 |
| Condition-paired | 5 speed conditions × 3 params | 15 |

### 4.2 Analysis 1: Correlation (FDR)

- Pearson r + Spearman ρ for all feature × target pairs
- FDR correction (Benjamini-Hochberg, q < 0.05)

### 4.3 Analysis 2: Regression

**ML (4)** — with permutation test (100 iter, best model only):

| Model | Tuning | Notes |
|-------|--------|-------|
| Ridge | RidgeCV (50 alphas) | L2 regularization |
| Lasso | LassoCV (50 alphas) | Sparse feature selection |
| ElasticNet | ElasticNetCV (L1+L2) | Balanced |
| XGBoost | GridSearchCV (depth, lr, reg) | Gradient boosting |

**DL (2–3)** — no permutation test:

| Model | Architecture | Input |
|-------|-------------|-------|
| MLP | 128→BN→ReLU→Drop(.5)→64→BN→ReLU→Drop(.5)→1 | Top-15 features |
| 1D-CNN | Conv(7→32,k5)→Conv(32→64,k5)→AvgPool→FC(64→32→1) | Raw PSD (7×79) |
| TabNet | n_d=8, n_a=8, n_steps=3 (optional) | Top-15 features |

**Dimensionality reduction (optional, --dim-reduce)**:
PCA+Ridge, SparsePCA+Ridge, PLS

### 4.4 Evaluation

- **Cross-validation**: 10-fold KFold (shuffle, seed=42)
- **Feature selection**: Top-15 by |Pearson r| within each fold (training only)
- **Metrics**: CV R², Spearman ρ, Pearson r, MAE
- **Permutation test**: 100 iterations, best ML model per target

### 4.5 Output

| File | Description |
|------|-------------|
| `output/correlation/correlation_fdr.csv` | All feature × target correlations |
| `output/correlation/correlation_report.md` | FDR summary |
| `output/regression/regression_results.csv` | All targets × models |
| `output/regression/regression_report.md` | Best models, SHAP, Lasso tracking |

---

## 5. Step 4 — RT Prediction (`analysis_rt.py`)

Predicts behavioral outcomes (RT_mean, RT_sd, ACC) per subject × condition.
Compares EEG-only vs EEG + task condition information.

### 5.1 Task Encoding

Structured 2d: Load (0/1) + log(Penalty)

### 5.2 Models (4 levels)

| Level | Model | Description |
|-------|-------|-------------|
| 1 | Ridge, XGBoost | Classical ML baselines |
| 2 | MLP | 128→64→3, LayerNorm + GELU |
| 3 | Neural SDE | Learned drift/diffusion with soft boundary |
| 4 | GNN + SDE | 7-ROI GCN encoder → Neural SDE |

### 5.3 Evaluation

- **CV**: 5-fold GroupKFold (subject-level split)
- **Comparison**: EEG only vs EEG + Task (×5 models = 10 experiments)
- **Metrics**: R², Spearman ρ, MAE

### 5.4 Output

| File | Description |
|------|-------------|
| `output/rt_prediction/rt_results.csv` | All models × input modes |
| `output/rt_prediction/rt_report.md` | Results + interpretation |

See `analysis/future_direction_deep_learning_ddm.md` for detailed design.

---

## 6. ROI Definitions

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

## 7. Execution

```bash
conda activate fastenv

# Step 1: Already done (193 subjects in raw_clean/)
# python code/preprocessing.py

# Step 2: Feature extraction (~15–20 min)
python code/extract_features.py

# Step 3: Feature–DDM analysis (~2 hours)
python code/analysis_features.py --dim-reduce

# Step 4: RT prediction (~25 min on GPU)
python code/analysis_rt.py --gnn
```

On cluster (SLURM):
```bash
sbatch submit.sh       # Steps 3
sbatch submit_rt.sh    # Step 4 (GPU required)
```

---

## 8. Key Design Decisions

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

## 9. Domain Knowledge Alignment

| Hypothesis | Evidence | Status |
|-----------|---------|--------|
| Aperiodic exponent → drift rate (v) | ★★★ (Euler 2024) | To verify with full features |
| IAF → processing speed | ★★ (Grandy 2013) | Now extracted |
| TBR → boundary separation (a) | ★★ (literature) | Prior: predicted v instead |
| Connectivity → DDM | Theoretical | **New: wPLI + coh included** |
| Entropy → DDM | Emerging | **New: sample + perm entropy** |
| Temporal ROI dominance | Unexpected prior finding | To investigate |

---

## 10. Code Files

| File | Purpose |
|------|---------|
| `preprocessing.py` | Raw .mff → clean epochs .fif |
| `extract_features.py` | Clean epochs → 231 EEG features + PSD matrix |
| `analysis_features.py` | EEG features ↔ DDM params (correlation + regression) |
| `analysis_rt.py` | EEG [+ task] → RT/ACC prediction (MLP, Neural SDE, GNN) |
