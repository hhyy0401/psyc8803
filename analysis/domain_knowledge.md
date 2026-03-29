# Domain Knowledge: Resting-State EEG Prediction of DDM Parameters

## Last Updated: 2026-03-25

---

## 1. Drift Diffusion Model (DDM) — Core Parameters

| Parameter | Symbol | Cognitive Meaning | Typical Range |
|-----------|--------|-------------------|---------------|
| Boundary separation | a | Response caution / speed-accuracy tradeoff | 0.5–2.0 |
| Drift rate | v | Evidence accumulation efficiency / information processing speed | -5.0–5.0 |
| Non-decision time | t0 (Ter) | Sensory encoding + motor execution time | 0.1–0.5 s |
| Starting point | z | Response bias (relative to boundaries) | ~a/2 (unbiased) |

- Larger **a** = more evidence required → slower but more accurate (conservative strategy)
- Higher **v** = faster, more efficient evidence accumulation → better discrimination
- **t0** = portion of RT unrelated to the decision process itself

---

## 2. Resting-State EEG Features — Taxonomy

### 2.1 Periodic (Oscillatory) Features

| Band | Frequency | Functional Significance |
|------|-----------|------------------------|
| Delta | 1–4 Hz | Deep sleep, cortical inhibition, homeostatic regulation |
| Theta | 4–8 Hz | Cognitive control (task-related ↑), drowsiness/mind-wandering (resting ↑) |
| Alpha | 8–13 Hz | Cortical idling, sensory gating, attentional suppression |
| Beta | 13–30 Hz | Active cortical processing, top-down inhibitory control, motor planning |
| Gamma | 30–40 Hz | Local cortical computation, perceptual binding, high-level processing |

**Key derived measures:**
- Theta/Beta Ratio (TBR): Marker of cognitive processing capacity (not just arousal); higher TBR = reduced executive control, mind-wandering, DMN engagement
- Individual Alpha Frequency (IAF/PAF): Temporal resolution of cortical processing; higher IAF = faster information processing
- Spectral entropy: Complexity/irregularity of spectral distribution

### 2.2 Aperiodic (1/f) Features (via FOOOF/specparam)

| Feature | Interpretation |
|---------|---------------|
| Exponent (slope) | Steepness of 1/f decay; reflects E/I balance. Steeper (larger exponent) = more inhibition-dominant, lower neural noise |
| Offset | Y-intercept in log-log space; reflects overall neural population spiking rate |

**Critical distinction:** FOOOF-parameterized periodic features (e.g., theta_peak_pw) represent *pure oscillatory power above the aperiodic floor*, whereas traditional absolute/relative band power confounds periodic and aperiodic components. This affects interpretation significantly.

---

## 3. Literature-Derived Hypotheses: EEG → DDM Mappings

### 3.1 Drift Rate (v) — Candidate Predictors

| Predictor | Direction | Mechanism | Evidence Strength | Key References |
|-----------|-----------|-----------|-------------------|----------------|
| Aperiodic exponent (slope) | Steeper → higher v | Lower neural noise → higher SNR → more efficient accumulation | ★★★ Strong | Euler et al. 2024 (Psychophysiology); Pathania et al. 2022 (Biol Psychol); Pi et al. 2024 (Psychophysiology) |
| Slow-wave power (delta/theta, raw) | Higher → lower v | Excess slow power = cortical hypoarousal → sluggish accumulation | ★★ Moderate | Kavcic et al. 2016 (only direct DDM study) |
| IAF / Peak Alpha Frequency | Higher → higher v | Faster temporal resolution → faster information sampling | ★★ Moderate (indirect) | Grandy et al. 2013 (NeuroImage); Finley et al. 2024 (J Neurosci) |
| Network efficiency (graph metrics) | Higher efficiency → higher v | More integrated networks → faster evidence propagation | ★★ Moderate (indirect) | Si et al. 2019 (J Neural Eng); Zhou et al. 2012 (Neuroscience) |
| Aperiodic offset | Higher → higher v (tentative) | Higher overall neural firing → greater computational throughput | ★ Weak | Implied by Manning et al. 2009; not directly tested for DDM |

### 3.2 Boundary Separation (a) — Candidate Predictors

| Predictor | Direction | Mechanism | Evidence Strength | Key References |
|-----------|-----------|-----------|-------------------|----------------|
| Theta/Beta Ratio | Higher TBR → lower a | Reduced executive control → impulsive (less conservative) responding | ★★ Moderate (indirect) | Schutte et al. 2017; Schutter & Van Honk 2005; Putman et al. 2010 |
| Frontal beta power | Higher → higher a | Top-down inhibitory control → higher response caution | ★ Weak (indirect) | Rogala et al. 2020 |
| Aperiodic exponent | Steeper → lower a (observed) | Better E/I balance → more confident, less need for caution? | ★ Novel finding | Current study data |
| Frontoparietal connectivity | Stronger → higher a | Stable strategy → conservative boundary setting | ★ Weak (indirect) | Rogala et al. 2020 |

### 3.3 Non-decision Time (t0) — Candidate Predictors

| Predictor | Direction | Mechanism | Evidence Strength |
|-----------|-----------|-----------|-------------------|
| Alpha power/frequency | Better alpha system → faster t0 | Sensory processing efficiency | ★ Weak (indirect) |
| Aperiodic offset | Higher → faster t0 (speculative) | Higher firing rate → faster encoding | ★ Speculative |

**Summary of literature maturity:**
- v: Reasonable empirical grounding, especially aperiodic slope
- a: Largely unexplored; TBR link is theoretically motivated but not directly tested for DDM
- t0: Almost entirely unexplored

---

## 4. Current Study: Empirical Findings (N = 184)

### 4.1 Feature Sets

| Feature Set | File | Features | Description |
|-------------|------|----------|-------------|
| Standard spectral | preprocess_v3.csv | 7 ROIs × 13 features = 91 | Absolute/relative band power, TBR, PAF, spectral entropy |
| FOOOF-parameterized | preprocess_v3_fooof.csv | 7 ROIs × 6 features = 42 | Aperiodic exponent/offset, periodic peak power per band (theta, alpha, beta, gamma) |

### 4.2 ROI Definitions

| ROI | Channels | Notes |
|-----|----------|-------|
| Frontal | E3, E6, E8, E9, E11, E2 | |
| Posterior | E34, E31, E40, E33, E38, E36 | |
| Central | E16, E7, E4, E54, E51, E41, E21 | |
| Left Temporal | E22, E24, E25, E26, E27 | Consistently top predictor ROI |
| Right Temporal | E49, E52, E48, E46, E45 | Consistently top predictor ROI |
| Occipital | E36, E37, E39, E32, E43 | |
| Prefrontal | E1, E17, E2, E11, E5, E10 | |

### 4.3 FDR-Significant Correlations (FOOOF features)

**9 correlations survived FDR correction — ALL aperiodic features:**

| DDM Variable | EEG Feature | r | FDR p | Direction |
|--------------|-------------|---|-------|-----------|
| Speed_Mid_v | right_temporal_aperiodic_exponent | +0.268 | 0.010 | Steeper slope → higher v ✓ |
| Accuracy_Max_a | left_temporal_aperiodic_exponent | -0.253 | 0.011 | Steeper slope → lower a |
| Accuracy_Max_a | right_temporal_aperiodic_exponent | -0.254 | 0.011 | Steeper slope → lower a |
| NoLoad_a | right_temporal_aperiodic_exponent | -0.264 | 0.012 | Steeper slope → lower a |
| Speed_Mid_v | posterior_aperiodic_offset | +0.240 | 0.015 | Higher offset → higher v |
| Speed_Mid_v | posterior_aperiodic_exponent | +0.239 | 0.015 | Steeper slope → higher v ✓ |
| Speed_Mid_v | right_temporal_aperiodic_offset | +0.234 | 0.015 | Higher offset → higher v |
| Grand Mean a | left_temporal_aperiodic_exponent | -0.229 | 0.037 | Steeper slope → lower a |
| Grand Mean a | right_temporal_aperiodic_exponent | -0.230 | 0.037 | Steeper slope → lower a |

### 4.4 Regression Prediction Performance (Best models)

| Target | Feature Set | Best Model | CV R² | Spearman ρ | Perm p | Note |
|--------|-------------|------------|-------|------------|--------|------|
| Speed_Mid_v | FOOOF | SVR | +0.044 | +0.218 | 0.005** | Best overall performance |
| Speed_Mid_v | Standard | ElasticNet | +0.041 | +0.161 | 0.005** | Comparable; TBR-driven |
| NoLoad_v | Standard | ElasticNet | +0.036 | +0.096 | 0.005** | TBR dominant features |
| NoLoad_a | Standard | RandomForest | +0.037 | +0.175 | 0.010* | Gamma/beta dominant |
| Neutral_a | Standard | Ridge | +0.025 | +0.081 | 0.020* | Gamma/beta dominant |
| Grand v | Standard | ElasticNet | +0.017 | +0.059 | 0.005** | |
| Grand a | FOOOF | Ridge | +0.009 | +0.079 | 0.065 | Marginal |

**Overall: CV R² consistently < 0.05; prediction is statistically detectable but practically very weak.**

### 4.5 Most Consistently Selected Features

**For boundary separation (a) — Standard features:**
- left_temporal_abs_gamma (10/10 folds, nearly all conditions)
- left_temporal_abs_beta, right_temporal_abs_beta
- central_abs_gamma, prefrontal_abs_gamma
- left_temporal_rel_gamma

**For boundary separation (a) — FOOOF features:**
- left_temporal_aperiodic_exponent (10/10)
- right_temporal_aperiodic_exponent (10/10)
- central_aperiodic_exponent (10/10)
- posterior/occipital_aperiodic_exponent

**For drift rate (v) — Standard features:**
- right_temporal_theta_beta_ratio (10/10 folds, nearly all conditions)
- central_theta_beta_ratio, left_temporal_theta_beta_ratio
- right_temporal_rel_theta
- left_temporal_rel_gamma

**For drift rate (v) — FOOOF features:**
- right_temporal_aperiodic_exponent (10/10)
- left_temporal_aperiodic_exponent (10/10)
- central_aperiodic_exponent (10/10)
- frontal_theta_peak_pw, right_temporal_theta_peak_pw

---

## 5. Alignment / Misalignment with Literature

### 5.1 Confirmed Alignments ✅

**(A) Aperiodic exponent → v (drift rate): STRONGLY ALIGNED**
- Literature predicts: steeper slope = lower neural noise = higher SNR → more efficient evidence accumulation
- Our data: positive correlation (r up to +0.27, FDR-sig), consistent across conditions
- Key supporting refs: Euler et al. 2024, Pathania et al. 2022, Pi et al. 2024
- This is the single strongest and most theoretically grounded finding

**(B) Aperiodic offset → v: PARTIALLY ALIGNED**
- Offset reflects neural spiking rate (Manning et al. 2009)
- Our data: positive correlation with Speed_Mid_v (r = +0.23–0.24, FDR-sig)
- Less explored in literature but mechanistically plausible (higher firing → more computational throughput)

**(C) Aperiodic exponent as domain-general cognitive predictor: ALIGNED**
- Euler et al. 2024: single general ability factor (g) best captures exponent-cognition link
- Finley et al. 2024: exponent predicts cognitive decline over 10 years
- Our data: exponent appears as top feature for BOTH a and v (different directions), consistent with domain-general role

### 5.2 Key Misalignments ❌

**(D) TBR → a (boundary separation): NOT CONFIRMED**
- Literature hypothesis: higher TBR → lower boundary (impulsive, risky choices) [Schutte 2017, Schutter & Van Honk 2005, Putman 2010]
- Our data: TBR is consistently selected as a predictor of **v, not a**
- Possible explanation: Recent reconceptualization of TBR as "cognitive processing capacity" marker rather than arousal/impulsivity (Picken et al. 2020, Clarke et al. 2019). If TBR reflects processing efficiency, its link to v (evidence accumulation efficiency) is more natural than to a (caution).
- **Implication:** The TBR → boundary separation link from decision-making literature may not directly translate to DDM parameter prediction from resting EEG.

**(E) Gamma power → a: NOVEL / UNEXPLAINED**
- No existing literature predicts high-frequency (gamma) power as a boundary separation predictor
- Our data: left_temporal_abs_gamma is the single most consistent feature for a across conditions (10/10 folds)
- Possible mechanism: Gamma reflects local cortical computation / sensory precision. Higher gamma might index greater perceptual confidence → lower need for response caution (lower a)?
- **Implication:** This is a novel empirical finding that requires theoretical development

**(F) Effect sizes are very small: IMPORTANT MISALIGNMENT with theoretical optimism**
- Literature paints a compelling theoretical picture of EEG → DDM links
- Reality: CV R² < 0.05, meaning resting EEG explains <5% of variance in DDM parameters
- This is consistent with the general finding that resting-state measures capture trait-level information weakly
- **Implication:** Resting EEG likely captures only a fraction of the factors determining individual DDM parameters

**(G) FOOOF theta_peak_pw direction for v: PARTIALLY MISALIGNED**
- Literature: raw slow-wave power ↑ → drift rate ↓ (Kavcic et al. 2016)
- Our FOOOF data: theta_peak_pw ↑ → v ↑ (positive direction)
- Resolution: FOOOF theta_peak_pw = pure oscillatory theta *above* the aperiodic floor, which is functionally distinct from raw theta power. Oscillatory theta may index active cognitive engagement rather than drowsiness.
- **Implication:** FOOOF parameterization changes the sign of theta–cognition relationships; raw vs. parameterized features are NOT interchangeable

**(H) Condition-specific predictions are unstable**
- Some conditions show reversed signs (e.g., Load_v: Lasso ρ = -0.26, perm p = 1.0)
- This suggests models overfit and that condition-specific DDM parameters have too much noise for stable resting-EEG prediction
- Grand mean or trait-level DDM parameters may be more appropriate targets

### 5.3 Spatial Specificity: Temporal ROI Dominance (NOVEL)

A striking pattern across both feature sets is the dominance of **bilateral temporal ROIs** (especially right temporal) as top predictors. This is unexpected because:
- Most aperiodic/cognition studies focus on fronto-central sites (e.g., Finley et al. 2024)
- Most TBR studies focus on frontal sites (Fz, F3, F4)

Possible explanations:
1. **Task-specific sensory processing:** If the DDM task involves visual or auditory stimuli, temporal cortex engagement may reflect individual differences in perceptual encoding
2. **Temporal E/I balance as trait marker:** Temporal cortex may have greater inter-individual variability in aperiodic properties
3. **Methodological artifact:** Temporal electrodes may be more susceptible to muscle artifact, potentially inflating feature importance
4. **Genuine finding:** Under-studied region that warrants attention in future DDM–EEG work

---

## 6. Key Reference Papers

### Tier 1: Directly relevant (aperiodic EEG → cognition/decision)

| Citation | Key Finding | Relevance |
|----------|-------------|-----------|
| Euler et al. 2024 (Psychophysiology) | Steeper resting aperiodic slope → faster RT, higher general cognitive ability (N=166) | Strongest support for exponent → v link |
| Pathania et al. 2022 (Biol Psychol) | Flatter frontal aperiodic slope → slower processing speed, poorer executive function | Supports exponent → v via processing speed |
| Pi et al. 2024 (Psychophysiology) | Higher aperiodic exponent → stable cognitive control style; lower exponent → compensatory shifts | Links exponent to control strategy (relevant to a) |
| Finley et al. 2024 (J Neurosci) | Fronto-central aperiodic exponent + IAPF jointly predict 10-year cognitive decline (N=235) | Longitudinal validation; exponent × IAPF interaction |
| Immink et al. 2021 (Acta Psychol) | Steeper 1/f slope → better visuomotor performance under demanding conditions | Performance prediction under load |

### Tier 2: DDM + EEG (mostly task-related, not resting)

| Citation | Key Finding | Relevance |
|----------|-------------|-----------|
| Nunez et al. 2017 (J Math Psychol) | Single-trial EEG (P200/N200) predicts drift rate and non-decision time | Methodological template for neural DDM |
| Nunez et al. 2024 (Behav Res Methods) | Tutorial on joint modeling of M/EEG and behavior with DDMs | Framework for integrating EEG into DDM estimation |
| Frank et al. 2015 (J Neurosci) | Mediofrontal theta modulates decision threshold; striatal activity modulates drift rate | fMRI+EEG → DDM; theta → boundary link |
| Kavcic et al. 2016 (Int J Psychophysiol) | Higher delta/theta power → lower drift rate in memory task | Only direct resting EEG → DDM (v) study |

### Tier 3: TBR and decision-making

| Citation | Key Finding | Relevance |
|----------|-------------|-----------|
| Schutte et al. 2017 (CABN) | Higher TBR → poorer reward-punishment reversal learning | TBR → flexible decision-making |
| Schutter & Van Honk 2005 (Cogn Brain Res) | Higher TBR → more disadvantageous high-risk choices (IGT) | TBR → risky decisions (→ lower a?) |
| Putman et al. 2010 (Biol Psychol) | Higher TBR → poorer fear-modulated response inhibition | TBR → inhibitory control |
| van Son et al. 2019 (Ann NY Acad Sci) | TBR covaries with mind-wandering and ECN connectivity | TBR as cognitive control state marker |
| Picken et al. 2020 (Clin EEG Neurosci) | TBR correlates with P300 latency (processing), not arousal markers | Reinterpretation: TBR = processing capacity |

### Tier 4: Network / alpha / other

| Citation | Key Finding | Relevance |
|----------|-------------|-----------|
| Torkamani-Azar et al. 2020 (IEEE JBHI) | Resting alpha features → RT and vigilance variability prediction | Alpha → decision speed |
| Si et al. 2019 (J Neural Eng) | Resting alpha network efficiency → higher acceptance rate (Ultimatum Game) | Network → decision tendency |
| Rogala et al. 2020 (Sci Rep) | Lower resting beta-2 (reconfigurable connectivity) → faster attention | Beta → flexible responding |
| Grandy et al. 2013 (NeuroImage) | Higher IAF → higher latent general cognitive ability factor | IAF → g → v (indirect) |

---

## 7. Future Research Directions

### 7.1 Methodological Improvements

#### (A) Integrated Feature Model: FOOOF + Standard
- Current analyses treat two feature sets independently
- Combining aperiodic parameters (exponent, offset) with oscillatory peaks and standard features may capture complementary variance
- Implementation: Run regression with both feature sets simultaneously; compare feature importance across sets
- Expected benefit: Aperiodic features → v (SNR mechanism) + TBR/gamma → a (control mechanism) may explain distinct variance

#### (B) Trait-Level DDM Targets
- Resting-state EEG is a trait-level measure; condition-specific DDM parameters include state-level noise
- Recommendation: Focus on **grand mean a and v** as primary targets
- Secondary targets: **condition contrasts** (e.g., Speed_a – Accuracy_a) which isolate strategic adjustment ability
- Condition-specific predictions should be presented as exploratory
- Rationale: Grand mean showed more consistent patterns; condition-specific targets showed sign reversals

#### (C) Feature Engineering
- **IAF (Individual Alpha Frequency)** is absent from current FOOOF feature set but strongly theorized as v predictor (Grandy et al. 2013). Extract from FOOOF alpha peak center frequency.
- **Cross-frequency coupling** (e.g., theta-gamma coupling) could capture integrative processing not reflected in single-band features
- **Asymmetry indices** (left vs. right temporal aperiodic exponent) could capture lateralized processing tendencies

#### (D) Addressing Low Prediction Power
- Current R² < 0.05 → explore non-linear feature interactions (e.g., exponent × IAF interaction shown by Finley et al. 2024)
- Consider stacking models or using PLS with more components
- Report effect sizes honestly; R² ~ 0.03–0.05 is consistent with typical resting-EEG → behavior correlations in the literature

### 7.2 Theoretical Extensions

#### (E) Gamma → Boundary Separation: Develop the Mechanism
- Novel finding: left temporal absolute gamma is the most consistent a predictor (standard features)
- Possible hypotheses to test:
  1. Gamma = perceptual precision → higher confidence → lower a
  2. Gamma = local E/I balance indicator (complementary to aperiodic exponent)
  3. Temporal gamma = auditory/multisensory readiness → task-specific effect
- How to test: Correlate gamma with other independent measures of perceptual confidence or impulsivity

#### (F) TBR as v (not a) Predictor: Reconceptualization
- Our data suggests TBR predicts v, contradicting the initial hypothesis (TBR → a)
- This aligns with the "TBR = cognitive processing capacity" view (Picken et al. 2020)
- Proposal: Frame TBR as reflecting baseline cognitive processing state, which manifests as evidence accumulation efficiency (v) rather than response caution strategy (a)
- Cite the ADHD literature's shift from "arousal" to "processing capacity" interpretation

#### (G) Temporal ROI Specificity
- Report and discuss the unexpected dominance of temporal (especially right temporal) ROI
- Compare with literature's focus on fronto-central sites
- May reflect genuine under-explored topography of DDM-relevant EEG signatures
- Consider task specifics: what kind of stimuli/response modality does your DDM task use?

### 7.3 Advanced Modeling Approaches

#### (H) Neural DDM / Joint Modeling
- Instead of two-step (fit DDM → correlate with EEG), embed EEG as covariates within hierarchical Bayesian DDM
- Framework: HDDM (Wiecki et al. 2013) or custom Stan model
- Resting EEG features as subject-level regressors on a, v, t0
- Benefits: Propagates uncertainty from DDM fitting into EEG-DDM relationship estimation; avoids double-dipping

#### (I) Multi-Modal / Multi-Scale Approach
- Combine resting EEG (trait-level) with task-related EEG (state-level) features
- Resting aperiodic exponent → trait v; task-evoked N200/CPP → trial-level v modulation
- This separates "baseline capacity" from "task engagement" contributions to decision-making

#### (J) Replication and Generalization
- N = 184 is moderate but not large for individual differences research
- Consider pre-registration of key hypotheses (especially exponent → v, exponent → a)
- Test generalizability across different DDM tasks (perceptual, value-based, social)
- Cross-dataset validation if possible

### 7.4 Specific Next Steps (Priority Order)

1. **Extract IAF** from FOOOF alpha peak and add to feature set
2. **Run integrated model** with both FOOOF + standard features
3. **Focus on grand mean a/v** as primary targets; relegate condition-specific to supplementary
4. **Test exponent × IAF interaction** (per Finley et al. 2024)
5. **Develop theoretical framing** for gamma → a and TBR → v findings
6. **Consider HDDM** with EEG covariates as advanced analysis

---

## 8. Key Interpretive Cautions

1. **FOOOF changes interpretation:** Raw theta power and FOOOF theta_peak_pw can have opposite correlational signs with cognition. Always specify which parameterization is used.

2. **Correlation ≠ Prediction:** Many nominally significant correlations (r ~ 0.15–0.20) do not translate to meaningful prediction (CV R² < 0.05). Report both.

3. **Multiple comparisons:** With 91 features × 16 DDM targets × 7 models, the total hypothesis space is enormous. FDR correction within each analysis helps but does not eliminate all false positives.

4. **Aperiodic exponent sign convention:** In FOOOF, the exponent is reported as positive (e.g., 1.5), representing the *steepness* of the 1/f decay. Larger exponent = steeper slope = more inhibition-dominant = lower neural noise. Some papers report the slope as negative (e.g., -1.5). Always clarify convention.

5. **Temporal ROI muscle artifact:** High-frequency features (gamma, beta) at temporal sites can be contaminated by temporalis muscle EMG. If gamma → a finding persists after aggressive artifact rejection, it strengthens the neural interpretation.

6. **Effect size expectations:** In resting EEG → individual differences literature, r ~ 0.15–0.25 is typical for well-powered studies. R² ~ 0.03–0.05 in cross-validated prediction is not a failure — it reflects the inherent ceiling of resting-state trait measures predicting specific cognitive parameters.
