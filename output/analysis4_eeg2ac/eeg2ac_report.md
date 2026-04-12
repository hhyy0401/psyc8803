## Composite Prediction (best model per target)

| Target | Model | R2 | Spearman | Perm p |
|--------|-------|----|----------|--------|
| WMC | PCA(3)+Ridge | -0.0020 | +0.0567 | - |
| gF | PCA(7)+Lasso | -0.0014 | +0.1251 | - |
| AC | PCA(5)+Lasso | +0.0730 | +0.2686 | - |
| SuS_AC | Ridge | -0.0129 | +0.1338 | 0.587 |

## EEG-AC FDR-significant correlations: 14

| Feature | r | q |
|---------|---|---|
| right_temporal_sample_entropy | -0.323 | 0.0008 |
| central_aperiodic_exponent | +0.279 | 0.0020 |
| central_sample_entropy | -0.282 | 0.0020 |
| right_temporal_aperiodic_exponent | +0.288 | 0.0020 |
| right_temporal_perm_entropy | -0.289 | 0.0020 |
| left_temporal_sample_entropy | -0.262 | 0.0041 |
| left_temporal_aperiodic_exponent | +0.253 | 0.0053 |
| frontal_aperiodic_exponent | +0.243 | 0.0066 |
| left_temporal_perm_entropy | -0.245 | 0.0066 |
| central_perm_entropy | -0.219 | 0.0176 |
| right_temporal_aperiodic_offset | +0.211 | 0.0218 |
| frontal_sample_entropy | -0.208 | 0.0230 |
| frontal_perm_entropy | -0.195 | 0.0352 |
| central_aperiodic_offset | +0.193 | 0.0352 |

## Individual Scores

| Target | Construct | R2 | Perm p |
|--------|-----------|-----|--------|
| Antisaccade | AC | +0.0248 | 0.015 |
| VAorient | AC | +0.0541 | 0.035 |
| Squared_Stroop | AC | +0.0262 | 0.015 |
| SART | SuS_AC | -0.0492 | 0.910 |
| SACT | SuS_AC | -0.0082 | 0.473 |
| PVT | SuS_AC | +0.0294 | 0.010 |
| SymSpan | WMC | -0.0176 | 0.687 |
| OSpan | WMC | -0.0169 | 0.687 |
| RotSpan | WMC | +0.0049 | 0.249 |
| RAPM | gF | +0.0883 | 0.005 |
| NumberSeries | gF | +0.0563 | 0.005 |
| LetterSets | gF | +0.0167 | 0.090 |