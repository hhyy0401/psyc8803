"""
Preprocess raw PSD (long-form) into ROI-grouped band power features.

Input:  all_subjects_raw_psd_long.csv  (subject, channel, frequency, power)
Output: preprocess_v1.csv              (one row per subject, ROI x band features)

Steps:
  1. Exclude QC-failed subjects (processing_failed == True in resting_subject_features.csv)
  2. Group channels into 7 ROIs
  3. Compute per-ROI: absolute & relative band power (delta, theta, alpha, beta, gamma),
     peak alpha frequency (PAF), theta/beta ratio, spectral entropy
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── ROI definitions ───
ROIS = {
    # ROI first attempt
    # "frontal":            ["E3", "E6", "E8", "E9"],
    # "posterior":          ["E26", "E34", "E45", "E35", "E37", "E38"],
    # "central":           ["E12", "E13", "E19", "E20", "E28", "E29", "E30", "E31"],
    # "left_temporal":     ["E14", "E15", "E21", "E22"],
    # "right_temporal":    ["E41", "E42", "E47", "E48"],
    # "parietal_extended": ["E39", "E40", "E43", "E44"],
    # "prefrontal":        ["E1", "E2", "E4", "E5", "E7", "E10"],

    "frontal":           ["E3", "E6", "E8", "E9"],
    "posterior":         ["E34", "E31", "E40", "E33", "E38", "E36"],
    "central":           ["E16", "E7", "E4", "E54", "E51", "E41", "E21"],
    "left_temporal":     ["E22", "E24", "E25", "E30"],
    "right_temporal":    ["E52", "E48", "E45", "E44"],
    "occipital":         ["E36","E37", "E39"],   # no parietal_extended
    "prefrontal":        ["E1", "E17", "E2", "E11", "E5", "E10"]
}

# ─── Band definitions (Hz) ───
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# ─── Load data ───
print("Loading raw PSD data...")
psd = pd.read_csv(os.path.join(base_dir, "all_subjects_raw_psd_long.csv"))
qc = pd.read_csv(os.path.join(base_dir, "resting_subject_features.csv"))

# ─── Exclude QC-failed subjects ───
failed_subjects = set(qc.loc[qc["processing_failed"] == True, "Subject"])
print(f"QC-failed subjects excluded: {sorted(failed_subjects)}")
psd = psd[~psd["subject"].isin(failed_subjects)].copy()

subjects = sorted(psd["subject"].unique())
print(f"Subjects remaining: {len(subjects)}")


# ─── Feature extraction ───
def spectral_entropy(power_values):
    """Normalized spectral entropy (0-1 range)."""
    p = np.array(power_values)
    p = p / p.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return entropy / np.log2(len(p)) if len(p) > 1 else 0.0


def extract_features(sub_roi_psd):
    """Extract band power features from a subject-ROI PSD (frequency, power)."""
    freqs = sub_roi_psd["frequency"].values
    powers = sub_roi_psd["power"].values

    total_power = np.trapz(powers, freqs)
    features = {}

    # Absolute and relative band power
    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            features[f"abs_{band_name}"] = np.nan
            features[f"rel_{band_name}"] = np.nan
            continue
        band_power = np.trapz(powers[mask], freqs[mask])
        features[f"abs_{band_name}"] = band_power
        features[f"rel_{band_name}"] = band_power / total_power if total_power > 0 else np.nan

    # Theta/beta ratio
    if features.get("abs_theta", 0) and features.get("abs_beta", 0):
        features["theta_beta_ratio"] = features["abs_theta"] / features["abs_beta"]
    else:
        features["theta_beta_ratio"] = np.nan

    # Peak alpha frequency (PAF)
    alpha_mask = (freqs >= 8.0) & (freqs < 13.0)
    if alpha_mask.sum() > 0:
        alpha_freqs = freqs[alpha_mask]
        alpha_powers = powers[alpha_mask]
        features["peak_alpha_freq"] = alpha_freqs[np.argmax(alpha_powers)]
    else:
        features["peak_alpha_freq"] = np.nan

    # Spectral entropy (full spectrum)
    features["spectral_entropy"] = spectral_entropy(powers)

    return features


print("Extracting features...")
rows = []
for subj in subjects:
    sub_psd = psd[psd["subject"] == subj]

    for roi_name, roi_channels in ROIS.items():
        roi_psd = sub_psd[sub_psd["channel"].isin(roi_channels)]
        if roi_psd.empty:
            continue

        # Average power across channels within ROI
        roi_avg = roi_psd.groupby("frequency")["power"].mean().reset_index()
        roi_avg = roi_avg.sort_values("frequency")

        feats = extract_features(roi_avg)
        row = {"subject": subj, "roi": roi_name}
        row.update(feats)
        rows.append(row)

features_long = pd.DataFrame(rows)

# ─── Pivot to wide format (one row per subject) ───
feature_cols = [c for c in features_long.columns if c not in ("subject", "roi")]
wide_parts = []
for roi_name in ROIS:
    roi_data = features_long[features_long["roi"] == roi_name][["subject"] + feature_cols].copy()
    roi_data = roi_data.rename(columns={c: f"{roi_name}_{c}" for c in feature_cols})
    roi_data = roi_data.set_index("subject")
    wide_parts.append(roi_data)

result = wide_parts[0]
for part in wide_parts[1:]:
    result = result.join(part, how="outer")

result = result.reset_index()

# ─── Save ───
output_path = os.path.join(base_dir, "preprocess_v2.csv")
result.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Shape: {result.shape} ({result.shape[0]} subjects x {result.shape[1]} features)")
print(f"Feature columns: {list(result.columns)}")
