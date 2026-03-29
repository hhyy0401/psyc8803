import os
import glob as _glob
import itertools
import numpy as np
import pandas as pd
import mne
import antropy as ant
from specparam import SpectralModel
from mne_connectivity import spectral_connectivity_epochs
from joblib import Parallel, delayed
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

try:
    from preprocessing import preprocess
except ImportError:
    preprocess = None  # only needed for raw .mff processing

# ================================
# Config
# ================================

BASE_DIR  = Path('/Users/hkim3239/GaTech Dropbox/Hyunju Kim/EEG')
CLEAN_DIR = str(BASE_DIR / 'raw_clean')
DATA_DIR  = BASE_DIR / 'data'
OUT_DIR   = BASE_DIR / 'output'
SAVE_PATH = str(DATA_DIR / 'resting_subject_features.csv')
N_JOBS    = 4   # parallel workers; lower if memory is tight

ROI_DEFS = {
    'frontal':        ['E3',  'E6',  'E8',  'E9',  'E11', 'E2'],
    'posterior':      ['E34', 'E31', 'E40', 'E33', 'E38', 'E36'],
    'central':        ['E16', 'E7',  'E4',  'E54', 'E51', 'E41', 'E21'],
    'left_temporal':  ['E22', 'E24', 'E25', 'E26', 'E27'],
    'right_temporal': ['E49', 'E52', 'E48', 'E46', 'E45'],
    'occipital':      ['E36', 'E37', 'E39', 'E32', 'E43'],
    'prefrontal':     ['E1',  'E17', 'E2',  'E11', 'E5',  'E10'],
}
ROI_NAMES = list(ROI_DEFS.keys())

BANDS = {
    'delta': (1,  4),
    'theta': (4,  7),
    'alpha': (8, 12),
    'beta':  (13, 30),
}
BAND_NAMES = list(BANDS.keys())

FOOOF_RANGE = [1, 40]
FOOOF_CFG   = dict(peak_width_limits=[1, 12], max_n_peaks=6, min_peak_height=0.1)
PSD_SCALE   = 1e12   # V² → pV² for FOOOF numerical stability


# ================================
# FOOOF per ROI
# ================================

def fit_fooof_roi(freqs, psd):
    """
    Fit FOOOF/specparam on one ROI's PSD.
    Returns dict of aperiodic params + per-band dominant peak (CF, PW, BW).
    """
    sm = SpectralModel(**FOOOF_CFG)
    sm.fit(freqs, psd * PSD_SCALE, FOOOF_RANGE)

    ap    = sm.get_params('aperiodic')   # [offset, exponent]
    peaks = sm.get_params('peak')        # (N, 3): [CF, PW, BW], or [] if no peaks

    result = {
        'aperiodic_offset':   float(ap[0]),
        'aperiodic_exponent': float(ap[1]),
    }
    for band, (flo, fhi) in BANDS.items():
        bp = peaks[(peaks[:, 0] >= flo) & (peaks[:, 0] < fhi)] if peaks.size > 0 else np.empty((0, 3))
        if len(bp) > 0:
            best = bp[np.argmax(bp[:, 1])]   # strongest peak in band
            result[f'{band}_peak_cf'] = float(best[0])
            result[f'{band}_peak_pw'] = float(best[1])
            result[f'{band}_peak_bw'] = float(best[2])
        else:
            result[f'{band}_peak_cf'] = np.nan
            result[f'{band}_peak_pw'] = 0.0
            result[f'{band}_peak_bw'] = np.nan

    return result


# ================================
# Full ROI-pair connectivity (wPLI, all pairs × all bands)
# ================================

def compute_full_connectivity(epochs_clean, roi_idx):
    """
    Compute wPLI between all 21 ROI pairs for delta/theta/alpha/beta using
    ROI-averaged virtual channels.  Returns 84 features.
    """
    data = epochs_clean.get_data()   # (n_epochs, n_channels, n_times)

    roi_data, valid_rois = [], []
    for roi in ROI_NAMES:
        idx = roi_idx[roi]
        if idx:
            roi_data.append(data[:, idx, :].mean(axis=1))
            valid_rois.append(roi)

    if len(valid_rois) < 2:
        return {}

    roi_data   = np.stack(roi_data, axis=1)   # (n_epochs, n_rois, n_times)
    info       = mne.create_info(valid_rois, epochs_clean.info['sfreq'], ch_types='eeg')
    roi_epochs = mne.EpochsArray(roi_data, info, verbose=False)

    con = spectral_connectivity_epochs(
        roi_epochs,
        method='wpli',
        mode='multitaper',
        fmin=[BANDS[b][0] for b in BAND_NAMES],
        fmax=[BANDS[b][1] for b in BAND_NAMES],
        faverage=True,
        verbose=False,
    )

    con_data = con.get_data(output='raveled')   # (n_pairs, n_bands)
    if con_data.ndim == 3:
        con_data = con_data.squeeze(-1)

    features = {}
    for ci, (ri, rj) in enumerate(itertools.combinations(range(len(valid_rois)), 2)):
        if ci >= con_data.shape[0]:
            break
        for bi, band in enumerate(BAND_NAMES):
            features[f'conn_{valid_rois[ri]}_{valid_rois[rj]}_{band}'] = float(con_data[ci, bi])

    return features


# ================================
# Entropy per ROI
# ================================

def compute_roi_entropy(epochs_clean, roi_idx):
    """
    Compute sample entropy and permutation entropy per ROI.
    Each metric is computed per 2-s epoch on the ROI-channel average,
    then averaged across epochs.
    Returns 14 features (7 ROIs × 2 entropy types).
    """
    data    = epochs_clean.get_data()   # (n_epochs, n_channels, n_times)
    result  = {}
    for roi in ROI_NAMES:
        idx = roi_idx[roi]
        if not idx:
            result[f'{roi}_sample_entropy'] = np.nan
            result[f'{roi}_perm_entropy']   = np.nan
            continue

        roi_signal = data[:, idx, :].mean(axis=1)   # (n_epochs, n_times)

        se = [ant.sample_entropy(ep) for ep in roi_signal]
        pe = [ant.perm_entropy(ep, normalize=True) for ep in roi_signal]

        result[f'{roi}_sample_entropy'] = float(np.nanmean(se))
        result[f'{roi}_perm_entropy']   = float(np.nanmean(pe))

    return result


# ================================
# Hemispheric asymmetry (FOOOF-based, L/R temporal)
# ================================

def compute_fooof_asymmetry(features):
    """
    Asymmetry index (R - L) / (|R| + |L|) for all FOOOF parameters
    between left_temporal and right_temporal.  Returns 14 features.
    """
    suffixes = ['aperiodic_offset', 'aperiodic_exponent']
    for band in BANDS:
        suffixes += [f'{band}_peak_cf', f'{band}_peak_pw', f'{band}_peak_bw']

    asym = {}
    for sf in suffixes:
        lv = features.get(f'left_temporal_{sf}', np.nan)
        rv = features.get(f'right_temporal_{sf}', np.nan)
        denom = abs(rv) + abs(lv)
        if denom > 0 and np.isfinite(lv) and np.isfinite(rv):
            asym[f'asym_{sf}'] = (rv - lv) / denom
        else:
            asym[f'asym_{sf}'] = np.nan

    return asym


# ================================
# Feature extraction
# ================================

def extract_features(epochs_clean, raw_ec):
    """
    Extract EEG features from preprocessed epochs.

    Parameters
    ----------
    epochs_clean : mne.Epochs
    raw_ec       : mne.Raw   (used for channel name lookup)

    Returns
    -------
    dict of feature name → scalar value
    """
    result = {}

    psd      = epochs_clean.compute_psd(fmin=1, fmax=40, verbose=False)
    freqs    = psd.freqs
    psd_mean = psd.get_data().mean(axis=0)   # (n_ch, n_freqs)

    ch_names = raw_ec.ch_names
    roi_idx  = {
        name: [ch_names.index(ch) for ch in chs if ch in ch_names]
        for name, chs in ROI_DEFS.items()
    }
    gl_idx = list(range(len(ch_names)))

    def band_power(ch_idx, fmin, fmax):
        idx = (freqs >= fmin) & (freqs <= fmax)
        return float(np.log(psd_mean[ch_idx][:, idx].mean()))

    # --- Band power ---
    result['posterior_alpha']          = band_power(roi_idx['posterior'],  8, 12)
    result['posterior_beta']           = band_power(roi_idx['posterior'],  13, 30)
    result['frontal_theta']            = band_power(roi_idx['frontal'],    4,  7)
    result['frontal_alpha']            = band_power(roi_idx['frontal'],    8, 12)
    result['frontal_theta_beta_ratio'] = (band_power(roi_idx['frontal'],   4,  7)
                                          - band_power(roi_idx['frontal'], 13, 30))
    result['prefrontal_theta']         = band_power(roi_idx['prefrontal'], 4,  7)
    result['central_alpha']            = band_power(roi_idx['central'],    8, 12)
    result['occipital_alpha']          = band_power(roi_idx['occipital'],  8, 12)
    result['global_delta']             = band_power(gl_idx, 1,  4)
    result['global_theta']             = band_power(gl_idx, 4,  7)
    result['global_alpha']             = band_power(gl_idx, 8, 12)
    result['global_beta']              = band_power(gl_idx, 13, 30)

    # --- IAF ---
    po_psd      = psd_mean[roi_idx['posterior']].mean(axis=0)
    alpha_range = (freqs >= 7) & (freqs <= 13)
    result['IAF'] = float(freqs[alpha_range][np.argmax(po_psd[alpha_range])])

    # --- Temporal alpha asymmetry (log-power based, kept for backward compat) ---
    if roi_idx['left_temporal'] and roi_idx['right_temporal']:
        result['temporal_alpha_asymmetry'] = (
            band_power(roi_idx['right_temporal'], 8, 12)
            - band_power(roi_idx['left_temporal'], 8, 12)
        )

    # --- FOOOF per ROI: aperiodic offset/exponent + per-band peak CF/PW/BW ---
    # Replaces the old global log-log linear fit with a proper specparam model
    # per ROI: 7 ROIs × 14 params = 98 features
    for roi in ROI_NAMES:
        idx = roi_idx[roi]
        if not idx:
            result[f'{roi}_aperiodic_offset']   = np.nan
            result[f'{roi}_aperiodic_exponent'] = np.nan
            for band in BANDS:
                result[f'{roi}_{band}_peak_cf'] = np.nan
                result[f'{roi}_{band}_peak_pw'] = 0.0
                result[f'{roi}_{band}_peak_bw'] = np.nan
            continue
        try:
            fooof_res = fit_fooof_roi(freqs, psd_mean[idx].mean(axis=0))
            for k, v in fooof_res.items():
                result[f'{roi}_{k}'] = v
        except Exception:
            result[f'{roi}_aperiodic_offset']   = np.nan
            result[f'{roi}_aperiodic_exponent'] = np.nan
            for band in BANDS:
                result[f'{roi}_{band}_peak_cf'] = np.nan
                result[f'{roi}_{band}_peak_pw'] = 0.0
                result[f'{roi}_{band}_peak_bw'] = np.nan

    # --- Hemispheric asymmetry (FOOOF-based, L/R temporal, 14 features) ---
    result.update(compute_fooof_asymmetry(result))

    # --- Connectivity: original specific pairs (wPLI + coherence, theta + alpha) ---
    fr_idx = np.array(roi_idx['frontal'])
    po_idx = np.array(roi_idx['posterior'])
    lt_idx = np.array(roi_idx['left_temporal'])
    rt_idx = np.array(roi_idx['right_temporal'])

    def compute_pair_connectivity(seeds, targets):
        s   = np.repeat(seeds, len(targets))
        t   = np.tile(targets, len(seeds))
        con = spectral_connectivity_epochs(
            epochs_clean,
            method=['wpli', 'coh'],
            indices=(s, t),
            fmin=(4,  8),
            fmax=(7, 12),
            faverage=True,
            verbose=False,
        )
        return con[0].get_data().mean(axis=0), con[1].get_data().mean(axis=0)

    fp_wpli, fp_coh = compute_pair_connectivity(fr_idx, po_idx)
    result['fp_theta_wpli'] = float(fp_wpli[0])
    result['fp_alpha_wpli'] = float(fp_wpli[1])
    result['fp_theta_coh']  = float(fp_coh[0])
    result['fp_alpha_coh']  = float(fp_coh[1])

    lr_wpli, lr_coh = compute_pair_connectivity(lt_idx, rt_idx)
    result['lr_theta_wpli'] = float(lr_wpli[0])
    result['lr_alpha_wpli'] = float(lr_wpli[1])
    result['lr_alpha_coh']  = float(lr_coh[1])

    # --- Full ROI-pair wPLI: all 21 pairs × 4 bands = 84 features ---
    result.update(compute_full_connectivity(epochs_clean, roi_idx))

    # --- Entropy per ROI: sample entropy + permutation entropy = 14 features ---
    result.update(compute_roi_entropy(epochs_clean, roi_idx))

    return result


# ================================
# Per-subject pipelines
# ================================

def process_one_subject(file_path, subject_id):
    """Pipeline from raw .mff file (requires preprocessing)."""
    try:
        epochs_clean, raw_ec, qc = preprocess(file_path)
        result = extract_features(epochs_clean, raw_ec)
        result.update(qc)
        result['Subject'] = subject_id
        result['qc_flag'] = (qc['n_bad_channels'] > 15) or (qc['n_epochs_kept'] < 100)
        print(f'[OK]   {subject_id}')
        return result
    except Exception as e:
        print(f'[FAIL] {subject_id}: {e}')
        return {'Subject': subject_id, 'processing_failed': True, 'error': str(e)}


def process_one_fif(fif_path, subject_id):
    """Pipeline from already-preprocessed *_clean-epo.fif file."""
    try:
        epochs_clean = mne.read_epochs(fif_path, preload=True, verbose=False)
        # epochs_clean serves as both epochs and channel-name source
        result = extract_features(epochs_clean, epochs_clean)
        result['Subject'] = subject_id
        result['n_epochs_kept'] = len(epochs_clean)
        print(f'[OK]   {subject_id}  ({len(epochs_clean)} epochs)')
        return result
    except Exception as e:
        print(f'[FAIL] {subject_id}: {e}')
        return {'Subject': subject_id, 'processing_failed': True, 'error': str(e)}


# ================================
# Main
# ================================

def compute_psd_matrix(fif_files, roi_defs, roi_names):
    """Load each .fif, compute ROI-averaged PSD, stack into matrix for CNN."""
    all_psd, subjects, ref_freqs = [], [], None

    for fif in fif_files:
        subj = int(os.path.basename(fif).split('_')[0])
        try:
            epochs = mne.read_epochs(fif, verbose=False)
            psd_obj = epochs.compute_psd(fmin=1, fmax=40, n_fft=500, verbose=False)
            freqs = psd_obj.freqs
            psd_mean = psd_obj.get_data().mean(axis=0)  # (n_ch, n_freqs)
            ch_names = epochs.ch_names

            roi_psd = []
            for roi in roi_names:
                idx = [i for i, ch in enumerate(ch_names) if ch in roi_defs[roi]]
                roi_psd.append(psd_mean[idx].mean(axis=0) if idx else np.full(len(freqs), np.nan))

            all_psd.append(np.stack(roi_psd))
            subjects.append(subj)
            if ref_freqs is None:
                ref_freqs = freqs
        except Exception:
            pass

    return np.stack(all_psd), np.array(subjects), ref_freqs


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'fooof').mkdir(exist_ok=True)

    fif_files = sorted(_glob.glob(os.path.join(CLEAN_DIR, '*_clean-epo.fif')))
    print(f'Found {len(fif_files)} preprocessed epoch files in {CLEAN_DIR}\n')

    jobs = [
        (fp, os.path.basename(fp).split('_')[0])   # e.g. '19248_clean-epo.fif' → '19248'
        for fp in fif_files
    ]

    all_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_one_fif)(fp, sid) for fp, sid in jobs
    )

    df = pd.DataFrame(all_results)
    df['Subject'] = pd.to_numeric(df['Subject'], errors='coerce')
    df = df.sort_values('Subject')

    # Save features.csv with 'subject' column (lowercase) for regression.py
    df_out = df.copy()
    df_out = df_out.rename(columns={'Subject': 'subject'})
    # Drop QC/meta columns — keep only numeric features
    meta_cols = ['processing_failed', 'error', 'qc_flag', 'n_bad_channels',
                 'n_epochs_total', 'n_epochs_kept', 'flag_low_epochs']
    feat_cols = [c for c in df_out.columns if c not in meta_cols]
    df_feat = df_out[feat_cols].dropna(subset=['subject'])
    df_feat.to_csv(SAVE_PATH, index=False)

    # Save PSD matrix for 1D-CNN
    print('\nComputing PSD matrix for CNN...')
    ok_fifs = [fp for fp, sid in jobs
               if int(sid) in df_feat['subject'].values]
    psd_matrix, psd_subjects, psd_freqs = compute_psd_matrix(
        ok_fifs, ROI_DEFS, ROI_NAMES
    )
    np.savez(
        OUT_DIR / 'psd_matrix.npz',
        psd=psd_matrix,
        subjects=psd_subjects,
        freqs=psd_freqs,
        roi_names=np.array(ROI_NAMES),
    )

    n_ok   = df['processing_failed'].isna().sum() if 'processing_failed' in df.columns else len(df)
    n_fail = df['processing_failed'].notna().sum() if 'processing_failed' in df.columns else 0
    print(f'\nDone. {n_ok} succeeded, {n_fail} failed.')
    print(f'Features:   {SAVE_PATH}  ({len(df_feat)} subjects × {len(feat_cols)-1} features)')
    print(f'PSD matrix: {OUT_DIR / "psd_matrix.npz"}  {psd_matrix.shape}')
