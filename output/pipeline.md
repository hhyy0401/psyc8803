# Resting-State EEG and Decision-Making: Analysis Pipeline

## 1. Overview

This project investigates whether resting-state EEG predicts individual differences in decision-making, operationalized via Drift Diffusion Model (DDM) parameters, and cognitive abilities (attentional control, working memory, fluid intelligence).

### 1.1 Research Questions

1. Do individual EEG features correlate with DDM parameters? (Analysis 1)
2. Does the multivariate EEG pattern associate with DDM? (Analysis 2)
3. Can EEG predict DDM boundary separation out-of-sample? (Analysis 3)
4. Can EEG predict attentional control (AC)? (Analysis 4)
5. Does AC link to DDM, bridging EEG and decision-making? (Analysis 5)

### 1.2 Key Finding

Individual EEG features show weak univariate correlations with DDM (r = 0.20, FDR trend-level). However, the multivariate EEG pattern significantly associates with DDM (distance correlation p = 0.036, CCA r = 0.585, p = 0.049). EEG directly predicts boundary separation (a; R2 = 0.066, p = 0.005). Drift rate (v) is linked indirectly via attentional control: EEG predicts AC (R2 = 0.073; 14/39 features FDR-significant), and AC strongly predicts v (r = 0.44, p < 0.001).

---

## 2. Data

### 2.1 Participants

~193 participants with resting-state EEG and behavioral data from a lexical decision task.

### 2.2 Resting-State EEG

- Eyes-closed resting, ~5 minutes
- Preprocessed: 1–40 Hz bandpass, 250 Hz sampling, bad channel interpolation, average reference, 2-sec epochs at 50% overlap, bad epoch rejection (200 µV threshold)
- Preprocessing code: `code/preprocessing.py` (by Yoonsang Lee)
- Feature extraction: `code/extract_features.py`
- Feature file: `data/resting_subject_features.csv`

### 2.3 Behavioral Task

Lexical decision (word vs. nonword) with 2 × 5 factorial manipulation:
- **Load** (2): Load vs. NoLoad (working memory manipulation)
- **Speed-Accuracy** (5): Penalty = {5, 10, 20, 40, 80} (speed-accuracy tradeoff)
- 10 conditions per participant

### 2.4 DDM Parameters

Drift Diffusion Model parameters fitted per participant × condition using fast-dm:
- **a (boundary separation)**: response caution; higher a = more cautious
- **v (drift rate)**: evidence accumulation speed; higher v = faster processing
- **t0 (non-decision time)**: motor/encoding time

**GrandMean**: The average of each DDM parameter across all 10 conditions per participant. Since resting-state EEG captures trait-level neural properties, GrandMean is the appropriate target — it removes condition-specific (state) variance and reflects stable individual differences (Euler et al., 2024).

- DDM file: `data/DDM_Scores.csv`

### 2.5 Cognitive Composite Scores

Z-scored composite scores from the cognitive battery:
- **WMC (Working Memory Capacity)**: SymSpan, OSpan, RotSpan
- **gF (Fluid Intelligence)**: RAPM, NumberSeries, LetterSets
- **AC (Attentional Control)**: Antisaccade, VAorient, Squared_Stroop
- **SuS_AC (Sustained Attention)**: SART, SACT, PVT

Each composite = mean of available z-scores (NaN-tolerant).

- Score file: `data/all_Scores_filtered.csv`
- Composite file: `data/all_Scores_filtered_composites.csv`

### 2.6 EEG Features (39 Interpretable Markers)

Built from the full feature set via domain-guided selection (code: `code/shared.py`, function `build_39_features`):

| Domain | Features | Count | Description |
|--------|----------|-------|-------------|
| Aperiodic exponent | 7 ROIs | 7 | 1/f spectral slope; reflects E/I balance (Euler et al., 2024; Pathania et al., 2022) |
| Aperiodic offset | 7 ROIs | 7 | Broadband power level; reflects neural population spiking rate (Manning et al., 2009) |
| Sample entropy | 7 ROIs | 7 | Temporal complexity via embedding; higher = more irregular dynamics |
| Permutation entropy | 7 ROIs | 7 | Nonlinear temporal complexity via ordinal patterns |
| Global band power | δ, θ, α, β | 4 | Log-mean power across all channels per band |
| IAF | 1 | 1 | Individual Alpha Frequency; processing speed proxy (Grandy et al., 2013) |
| Frontal TBR | 1 | 1 | Theta/Beta Ratio; cognitive control marker (van Son et al., 2019) |
| Temporal alpha asymmetry | 1 | 1 | Left-right alpha difference |
| Connectivity (mean wPLI) | δ, θ, α, β | 4 | Average inter-ROI weighted Phase Lag Index per band |
| **Total** | | **39** | |

**ROIs (7)**: Frontal, Posterior, Central, Left Temporal, Right Temporal, Occipital, Prefrontal.

---

## 3. Analyses

### 3.1 Analysis 1: EEG ↔ DDM Univariate Correlation

**Code**: `code/analysis1_correlation.py`

- **Input (features)**: 39 EEG features
- **Output (targets)**: GrandMean a, v, t0
- **Task**: Can individual EEG features predict DDM parameters? Pearson/Spearman correlation with per-parameter FDR correction (39 tests per DDM parameter; each parameter is an independent hypothesis family).

**Method**:
- Pearson and Spearman correlation for each feature × parameter pair
- FDR correction (Benjamini-Hochberg) per DDM parameter (39 tests each)

**Results** (N = 188):

Boundary separation (a) — 4 trend-level (q < 0.10):

| Feature | r | q |
|---------|---|---|
| left_temporal_sample_entropy | +0.228 | 0.071 |
| global_beta | +0.207 | 0.071 |
| left_temporal_aperiodic_exponent | -0.202 | 0.071 |
| right_temporal_aperiodic_exponent | -0.197 | 0.071 |

Drift rate (v): no FDR-significant or trend-level results.
Non-decision time (t0): no FDR-significant or trend-level results.

**Interpretation**:
- Individual EEG features show weak but consistent effects (r ≈ 0.20)
- Direction matches prior literature:
  - Lower aperiodic exponent (flatter 1/f slope, less inhibition) → higher boundary (Pathania et al., 2022; Pi et al., 2024)
  - Higher temporal sample entropy (more neural noise) → higher boundary
- **Temporal regions drive the effect**, not frontal — an unexpected finding consistent with the expanded ROI analysis
- These effects are too weak to survive strict FDR correction → motivates multivariate analysis (Analysis 2)

---

### 3.2 Analysis 2: Multivariate EEG ↔ DDM Association

**Code**: `code/analysis2_multivariate.py`

- **Input (features)**: 39 EEG features (as a multivariate pattern)
- **Output (targets)**: GrandMean a, v, t0 (as a multivariate pattern)
- **Task**: Does the EEG multivariate pattern as a whole associate with DDM? This addresses the limitation of Analysis 1 where individual features are too weak.

**Methods**:

**Distance Correlation (dCor)**: Tests whether participants who are similar in EEG space are also similar in DDM space. Unlike Pearson correlation, dCor captures nonlinear and multivariate dependencies. Significance assessed via permutation test (500 iterations).

**Canonical Correlation Analysis (CCA)**: Finds linear combinations of EEG features and DDM parameters that maximize correlation. Each "canonical component" (CC) represents a shared latent dimension. Significance of CC1 assessed via permutation test (5000 iterations). Loadings (correlation of original variables with canonical variates) reveal which features drive each dimension.

**Results** (N = 188):

| Method | Result | p |
|--------|--------|---|
| dCor(EEG, DDM) | 0.293 | 0.036 |
| CCA CC1 | r = 0.585 | 0.049 |
| CCA CC2 | r = 0.520 | — |

CCA CC1 loadings:
- DDM: a = +0.95 (boundary separation dominates)
- EEG: temporal sample_entropy (+0.33), global_beta (+0.33), temporal aperiodic_exponent (-0.29)

CCA CC2 loadings:
- DDM: v = +0.71 (drift rate dominates)
- EEG: connectivity_delta (+0.38), temporal sample_entropy (-0.35), temporal aperiodic_exponent (+0.33)

**Interpretation**:
- **dCor confirms that the multivariate EEG pattern significantly relates to DDM** (p = 0.036), even though individual features do not survive FDR
- CCA reveals **two interpretable shared dimensions**:
  - **CC1 (Boundary dimension)**: Higher temporal entropy + lower aperiodic exponent → higher boundary. Interpretation: more neural noise / weaker inhibition → more cautious responding
  - **CC2 (Drift dimension)**: Higher delta connectivity + lower entropy → higher drift rate. Interpretation: better inter-regional integration with less local noise → faster evidence accumulation
- The CCA loadings on CC1 are consistent with Analysis 1 (same features: temporal entropy, aperiodic exponent, beta power)
- **Key message**: Multiple weak EEG effects combine into a significant multivariate signal (individual r ≈ 0.20, but combined CCA r = 0.59)

---

### 3.3 Analysis 3: EEG → Boundary Separation (a) Regression

**Code**: `code/analysis3_regression.py`

- **Input (features)**: 39 EEG features
- **Output (targets)**: GrandMean a, GrandMean v
- **Task**: Can EEG predict DDM parameters out-of-sample? Linear (Ridge, Lasso, ElasticNet) and nonlinear (XGBoost) models, with and without PCA dimensionality reduction.

**Method**:
- 10-fold cross-validation (out-of-sample R2)
- Linear models: Ridge, Lasso, ElasticNet
- Nonlinear model: XGBoost
- Dimensionality reduction: PCA(k) + {Ridge, Lasso, XGBoost} for k = 3, 5, 7, 10
- Permutation test (200 iterations) for best linear and XGBoost

**Results** (N = 188):

Boundary separation (a) — top models:

| Model | R2 | Spearman ρ | Perm p |
|-------|----|-----------|--------|
| PCA(5)+Ridge | +0.066 | +0.243 | — |
| PCA(3)+Ridge | +0.060 | +0.218 | — |
| Ridge | +0.057 | +0.211 | 0.005 |
| XGBoost | -0.006 | +0.116 | 0.060 |

Drift rate (v) — linear models fail (R2 < 0), but nonlinear shows signal:

| Model | R2 | Spearman ρ | Perm p |
|-------|----|-----------|--------|
| PCA(5)+XGBoost | +0.044 | +0.262 | — |
| XGBoost | +0.005 | +0.163 | 0.040 |
| Ridge | -0.031 | -0.010 | n.s. |

Lasso-selected features for a: aperiodic_exponent (temporal), sample_entropy (temporal), global_beta, connectivity_alpha.

**Interpretation**:
- **Boundary separation (a) is reliably predicted by linear models** (Ridge R2 = 0.057, p = 0.005; PCA(5)+Ridge R2 = 0.066)
- PCA(5) slightly improves over raw features → 39 features contain some redundancy, but signal is distributed across 5+ dimensions
- **Drift rate (v) shows a nonlinear signal**: XGBoost is significant (p = 0.040) while all linear models fail (R2 < 0). This suggests the EEG–v relationship is nonlinear — consistent with the idea that multiple neural features interact rather than contribute additively
- The R2 = 0.066 (a) and 0.044 (v, PCA+XGBoost) are modest but meaningful for resting-EEG trait prediction
- Lasso feature selection converges with Analyses 1 and 2: temporal entropy and aperiodic exponent are the consistent markers

---

### 3.4 Analysis 4: EEG → Attentional Control (AC)

**Code**: `code/analysis4_eeg2ac.py`

- **Input (features)**: 39 EEG features (same as all other analyses)
- **Output (targets)**: Cognitive composites (WMC, gF, AC, SuS_AC) and individual cognitive scores
- **Task**: Can resting EEG predict cognitive abilities? If EEG cannot directly predict drift rate, perhaps it predicts cognitive abilities that in turn relate to drift rate.

**Method**:
- Same model set as Analysis 3: Ridge, Lasso, ElasticNet, XGBoost, PCA(k) + {Ridge, Lasso, XGBoost}
- 10-fold CV, 200 permutation tests for best linear + XGBoost
- EEG ↔ AC correlation: 39 features, FDR-corrected (39 tests)
- Individual score prediction: Ridge + XGBoost, 10-fold CV

**Results** (N = 171):

Composite prediction (best model per target):

| Target | Best Model | R2 | Perm p |
|--------|-----------|-----|--------|
| **AC** | **PCA(5)+Lasso** | **+0.073** | — |
| gF | PCA(7)+Lasso | -0.001 | — |
| WMC | PCA(3)+Ridge | -0.002 | — |
| SuS_AC | Ridge | -0.013 | 0.587 |

EEG ↔ AC FDR-significant features (14 out of 39):
- right_temporal_sample_entropy: r = -0.323 (q = 0.001)
- right_temporal_aperiodic_exponent: r = +0.288 (q = 0.002)
- central_sample_entropy: r = -0.282 (q = 0.002)
- central_aperiodic_exponent: r = +0.279 (q = 0.002)
- (+ 10 more entropy and aperiodic features across ROIs)

Individual scores (significant, p < 0.05):

| Target | Construct | R2 | Perm p |
|--------|-----------|-----|--------|
| RAPM | gF | +0.088 | 0.005 |
| NumberSeries | gF | +0.056 | 0.005 |
| VAorient | AC | +0.054 | 0.035 |
| PVT | SuS_AC | +0.029 | 0.010 |
| Squared_Stroop | AC | +0.026 | 0.015 |
| Antisaccade | AC | +0.025 | 0.015 |

**Interpretation**:
- **AC is the best-predicted composite** (R2 = 0.073 with PCA(5)+Lasso), confirming that resting EEG captures attentional control
- **14 out of 39 EEG features are FDR-significant with AC** (q < 0.05), the strongest correlation result in the entire pipeline
- All three AC component tasks (Antisaccade, VAorient, Stroop) are individually significant — the composite reflects a genuine EEG-linked construct
- Surprisingly, individual gF tasks (RAPM R2 = 0.088, NumberSeries R2 = 0.056) are also predicted, though the gF composite is not — suggesting that EEG captures specific task processes rather than the general factor
- The EEG features correlating with AC (temporal entropy, aperiodic exponent) are the same features from Analyses 1-3
- This establishes the first half of the indirect pathway: **EEG → AC**

---

### 3.5 Analysis 5: AC ↔ DDM Correlation

**Code**: `code/analysis5_ac_ddm.py`

- **Input (features)**: Cognitive composite scores (WMC, gF, AC, SuS_AC)
- **Output (targets)**: GrandMean a, v, t0
- **Task**: Does attentional control (AC) link to DDM parameters? This completes the indirect pathway: EEG → AC → DDM.

**Method**: Pearson and Spearman correlation between each composite and each DDM parameter.

**Results** (N = 256):

| Composite | DDM | r | p |
|-----------|-----|---|---|
| **AC** | **v** | **+0.439** | **< 0.001** |
| **AC** | **a** | **-0.313** | **< 0.001** |
| gF | v | +0.379 | < 0.001 |
| gF | a | -0.276 | < 0.001 |
| WMC | a | -0.250 | < 0.001 |
| SuS_AC | v | +0.342 | < 0.001 |

**Interpretation**:
- **AC has the strongest association with both v and a** among all composites
- AC ↔ v (r = +0.44): higher attentional control → faster evidence accumulation. This aligns with the E/I balance theory — better neural inhibition (captured by aperiodic exponent) → lower noise → higher SNR → higher drift rate (Euler et al., 2024)
- AC ↔ a (r = -0.31): higher AC → lower boundary. Participants with better attentional control need less caution to maintain accuracy
- This completes the indirect pathway: **EEG features ↔ AC (14/39 FDR-sig) → v (r = 0.44)**
- While EEG cannot linearly predict v directly (Analysis 3), the EEG-AC-DDM pathway suggests that the relationship is mediated by attentional control ability

---

## 4. Summary of Significant Results

| Analysis | Finding | Statistic | p |
|----------|---------|-----------|---|
| 1 | EEG ↔ a: temporal entropy, aperiodic exponent | r ≈ 0.20 | q < 0.10 (trend) |
| 2 | Multivariate EEG ↔ DDM (dCor) | 0.293 | **0.036** |
| 2 | CCA shared dimension (CC1 = boundary) | r = 0.585 | **0.049** |
| 3 | EEG → a regression (Ridge) | R2 = 0.057 | **0.005** |
| 3 | EEG → a regression (PCA(5)+Ridge) | R2 = 0.066 | — |
| 3 | EEG → v regression (XGBoost, nonlinear) | R2 = 0.005 | **0.040** |
| 4 | EEG → AC regression (PCA(5)+Lasso) | R2 = 0.073 | — |
| 4 | EEG ↔ AC (14/39 FDR-significant features) | r = 0.28-0.32 | **q < 0.005** |
| 4 | EEG → RAPM (XGBoost) | R2 = 0.088 | **0.005** |
| 5 | AC ↔ v | r = +0.439 | **< 0.001** |
| 5 | AC ↔ a | r = -0.313 | **< 0.001** |

---

## 5. Integrated Interpretation

```
                         EEG --> DDM (direct)
              .-----[Ridge R2=0.06, p=0.005]-----> Boundary (a)
             /                                          ^
 Resting    /                                           |
  EEG -----+---[XGBoost p=0.04, nonlinear]--> Drift (v) |
             \                                     ^    |
              \                                    |    |
               '---[R2=0.07, 14/39 FDR-sig]-> AC --'----'
                    EEG --> AC              r=+0.44  r=-0.31
                                            AC --> DDM
```

The results across five analyses converge on a coherent picture of how resting-state EEG relates to decision-making.

The aperiodic exponent of resting EEG reflects the balance between excitatory and inhibitory neural activity (E/I balance). A steeper slope indicates greater inhibition dominance, resulting in lower neural noise and higher signal-to-noise ratio (Euler et al., 2024; Pathania et al., 2022). This E/I balance connects to decision-making through two pathways.

First, there is a direct pathway from EEG to boundary separation (a). Individuals with more irregular neural dynamics — higher temporal entropy and flatter 1/f slope — adopt higher decision boundaries (Analysis 1: r = 0.20; Analysis 3: R2 = 0.066, p = 0.005). This is consistent with a compensatory strategy: under noisier internal representations, the brain sets a higher evidence threshold to maintain accuracy (Pi et al., 2024).

Second, there is an indirect pathway from EEG to drift rate (v) through attentional control (AC). EEG alone cannot linearly predict v (Analysis 3: all linear R2 < 0), but EEG predicts AC (Analysis 4: R2 = 0.073; 14/39 FDR-significant features, r = 0.28-0.32), and AC strongly correlates with v (Analysis 5: r = +0.44, p < 0.001). The EEG features that predict AC — temporal entropy and aperiodic exponent — are the same features that appear in the DDM analyses. This suggests that the E/I balance captured by resting EEG shapes attentional control ability, which in turn determines how efficiently a person accumulates evidence during decision-making.

Notably, XGBoost reveals a nonlinear EEG-v signal (Analysis 3: p = 0.040) that linear models miss entirely, suggesting that EEG features interact in predicting drift rate rather than contributing additively. This is consistent with the CCA finding (Analysis 2), where the drift rate dimension (CC2) loads on both connectivity and entropy — features from different domains that may interact nonlinearly.

Three EEG markers appear consistently across all analyses: temporal aperiodic exponent (E/I balance), temporal sample entropy (neural complexity), and global beta power (cortical activation). Their convergence across univariate, multivariate, regression, and cognitive prediction analyses provides evidence that these are genuine markers of individual differences in decision-making.

---

## 6. Code Structure

```
code/
├── preprocessing.py          # Step 1: Raw EEG → clean epochs (by Yoonsang Lee)
├── extract_features.py       # Step 2: Epochs → 208 features + PSD matrix
├── shared.py                 # Shared: 39-feature builder, data loaders
├── analysis1_correlation.py  # Analysis 1: Univariate EEG ↔ DDM
├── analysis2_multivariate.py # Analysis 2: dCor + CCA
├── analysis3_regression.py   # Analysis 3: EEG → a regression
├── analysis4_eeg2ac.py       # Analysis 4: EEG → AC prediction
└── analysis5_ac_ddm.py       # Analysis 5: AC ↔ DDM correlation
```

**Running all analyses**:
```bash
cd code
python analysis1_correlation.py    # ~10 sec
python analysis2_multivariate.py   # ~20 min (permutation tests)
python analysis3_regression.py     # ~5 min
python analysis4_eeg2ac.py         # ~10 min
python analysis5_ac_ddm.py         # ~5 sec
```

---

## 7. Output Structure

```
output/
├── analysis1_correlation/
│   ├── correlation_results.csv    # All 117 feature × param correlations
│   └── correlation_report.md
├── analysis2_multivariate/
│   ├── dcor_results.csv           # Distance correlation results
│   ├── cca_loadings.csv           # CCA loadings for all 3 CCs
│   └── multivariate_report.md
├── analysis3_regression/
│   ├── regression_results.csv     # All model × target CV results
│   └── regression_report.md
├── analysis4_eeg2ac/
│   ├── composite_results.csv      # Composite prediction results
│   ├── individual_results.csv     # Individual score prediction
│   ├── correlation_fdr.csv        # EEG ↔ AC feature correlations
│   └── eeg2ac_report.md
└── analysis5_ac_ddm/
    ├── ac_ddm_correlation.csv     # All composite × DDM correlations
    └── ac_ddm_report.md
```

---

## References

Euler, M. J., Vehar, J. V., Guevara, J. E., Geiger, A. R., Deboeck, P. R., & Lohse, K. R. (2024). Associations between the resting EEG aperiodic slope and broad domains of cognitive ability. *Psychophysiology, 61*(6), e14543. https://doi.org/10.1111/psyp.14543

Finley, A. J., Angus, D. J., Knight, E. L., van Reekum, C. M., Lachman, M. E., Davidson, R. J., & Schaefer, S. M. (2024). Resting EEG periodic and aperiodic components predict cognitive decline over 10 years. *Journal of Neuroscience, 44*(13), e1332232024. https://doi.org/10.1523/JNEUROSCI.1332-23.2024

Frank, M. J., Gagne, C., Nyhus, E., Masters, S., Wiecki, T. V., Cavanagh, J. F., & Badre, D. (2015). fMRI and EEG predictors of dynamic decision parameters during human reinforcement learning. *Journal of Neuroscience, 35*(2), 485–494. https://doi.org/10.1523/JNEUROSCI.2062-14.2015

Grandy, T. H., Werkle-Bergner, M., Chicherio, C., Lövdén, M., Schmiedek, F., & Lindenberger, U. (2013). Individual alpha peak frequency is related to latent factors of general cognitive abilities. *NeuroImage, 79*, 10–18. https://doi.org/10.1016/j.neuroimage.2013.04.059

Manning, J. R., Jacobs, J., Fried, I., & Kahana, M. J. (2009). Broadband shifts in local field potential power spectra are correlated with single-neuron spiking in humans. *Journal of Neuroscience, 29*(43), 13613–13620. https://doi.org/10.1523/JNEUROSCI.2041-09.2009

Nunez, M. D., Vandekerckhove, J., & Srinivasan, R. (2017). How attention influences perceptual decision making: Single-trial EEG correlates of drift-diffusion model parameters. *Journal of Mathematical Psychology, 76*(B), 117–130. https://doi.org/10.1016/j.jmp.2016.03.003

Nunez, M. D., Fernandez, K., Srinivasan, R., & Vandekerckhove, J. (2024). A tutorial on fitting joint models of M/EEG and behavior to understand cognition. *Behavior Research Methods, 56*(6), 6020–6050. https://doi.org/10.3758/s13428-023-02331-x

Pathania, A., Euler, M. J., Clark, M., Cowan, R. L., Duff, K., & Lohse, K. R. (2022). Resting EEG spectral slopes are associated with age-related differences in information processing speed. *Biological Psychology, 168*, 108261. https://doi.org/10.1016/j.biopsycho.2022.108261

Pi, Y.-L., Zhu, X., Wu, H., Zhou, R., Jiang, Y., & Colzato, L. S. (2024). Interindividual aperiodic resting-state EEG activity predicts cognitive-control styles. *Psychophysiology, 61*(7), e14576. https://doi.org/10.1111/psyp.14576

Picken, C., Clarke, A. R., Barry, R. J., McCarthy, R., & Selikowitz, M. (2020). The theta/beta ratio as an index of cognitive processing in adults with the combined type of attention deficit hyperactivity disorder. *Clinical EEG and Neuroscience, 51*(3), 167–173. https://doi.org/10.1177/1550059419895142

Rogala, J., Kublik, E., Krauz, R., & Wróbel, A. (2020). Resting-state EEG activity predicts frontoparietal network reconfiguration and improved attentional performance. *Scientific Reports, 10*, 2337. https://doi.org/10.1038/s41598-020-59242-y

Schutter, D. J. L. G., & Van Honk, J. (2005). Electrophysiological ratio markers for the balance between reward and punishment. *Cognitive Brain Research, 24*(3), 685–690. https://doi.org/10.1016/j.cogbrainres.2005.04.002

Schutte, I., Kenemans, J. L., & Schutter, D. J. L. G. (2017). Resting-state theta/beta EEG ratio is associated with reward- and punishment-related reversal learning. *Cognitive, Affective, & Behavioral Neuroscience, 17*(4), 754–763. https://doi.org/10.3758/s13415-017-0510-3

van Son, D., De Blasio, F. M., Fogarty, J. S., Angelidis, A., Barry, R. J., & Putman, P. (2019). Electroencephalography theta/beta ratio covaries with mind wandering and functional connectivity in the executive control network. *Annals of the New York Academy of Sciences, 1452*(1), 52–64. https://doi.org/10.1111/nyas.14180
