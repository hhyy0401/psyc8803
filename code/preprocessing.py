import os
import numpy as np
import mne
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

# ================================
# Config
# ================================

BASE_DIR  = '/Users/hkim3239/GaTech Dropbox/Hyunju Kim/EEG'
BASE_PATH = os.path.join(BASE_DIR, 'raw')
TXT_PATH  = os.path.join(BASE_DIR, 'data', 'resting_usable_list.txt')
CLEAN_DIR = os.path.join(BASE_DIR, 'raw_clean')
N_JOBS    = 4   # parallel workers; lower if memory is tight


# ================================
# Helpers
# ================================

def crop_ec_segment(raw, discard_sec=10):
    bad_skips = [
        (onset, onset + duration)
        for onset, duration, desc in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
        if desc == 'BAD_ACQ_SKIP'
    ]
    if len(bad_skips) < 2:
        raise ValueError('Not enough BAD_ACQ_SKIP markers to define EC segment.')
    ec_start = bad_skips[1][1] + discard_sec
    return raw.copy().crop(tmin=ec_start)


# ================================
# Preprocessing
# ================================

def preprocess(file_path):
    """
    Load and preprocess one EGI resting-state file.

    Returns
    -------
    epochs_clean : mne.Epochs
    raw_ec       : mne.Raw   (post-reference, for channel name lookup)
    qc           : dict      (n_bad_channels, n_epochs_total, n_epochs_kept, flag_low_epochs)
    """
    qc = {}

    # Load & crop
    raw    = mne.io.read_raw_egi(file_path, preload=True, verbose=False)
    raw_ec = crop_ec_segment(raw, discard_sec=10)
    raw_ec.pick_channels([ch for ch in raw_ec.ch_names if ch.startswith('E')])

    # Resample + filter
    raw_ec.resample(250, verbose=False)
    raw_ec.notch_filter(60, verbose=False)
    raw_ec.filter(1.0, 40, verbose=False)

    # Bad channel detection
    data_uv    = raw_ec.get_data() * 1e6
    stds       = np.std(data_uv, axis=1)
    median_std = np.median(stds)

    var_bad  = [ch for i, ch in enumerate(raw_ec.ch_names) if stds[i] > 5 * median_std]

    psd_line = raw_ec.compute_psd(fmin=55, fmax=65, verbose=False)
    lp       = psd_line.get_data()[:, np.argmin(np.abs(psd_line.freqs - 60))]
    line_bad = [ch for i, ch in enumerate(raw_ec.ch_names)
                if lp[i] > np.median(lp) + 4 * np.std(lp)]

    final_bad = sorted(set(var_bad + line_bad))
    qc['n_bad_channels'] = len(final_bad)

    raw_ec.info['bads'] = final_bad
    raw_ec.interpolate_bads(reset_bads=True)
    raw_ec.set_eeg_reference('average', verbose=False)

    # Epoch
    epochs = mne.make_fixed_length_epochs(
        raw_ec, duration=2.0, overlap=1.0, preload=True, verbose=False
    )
    qc['n_epochs_total'] = len(epochs)

    epochs_clean = epochs.copy().drop_bad(reject=dict(eeg=200e-6))
    qc['n_epochs_kept']   = len(epochs_clean)
    qc['flag_low_epochs'] = qc['n_epochs_kept'] < 100

    return epochs_clean, raw_ec, qc


# ================================
# Save one subject
# ================================

def preprocess_and_save(file_path, subject_id, clean_dir):
    save_path = os.path.join(clean_dir, f'{subject_id}_clean-epo.fif')

    if os.path.exists(save_path):
        print(f'[SKIP] {subject_id} already exists')
        return

    try:
        epochs_clean, _, qc = preprocess(file_path)
        epochs_clean.save(save_path, overwrite=True, verbose=False)
        print(f'[OK]   {subject_id}  '
              f'(bad_ch={qc["n_bad_channels"]}, epochs={qc["n_epochs_kept"]})')
    except Exception as e:
        print(f'[FAIL] {subject_id}: {e}')


# ================================
# Main
# ================================

if __name__ == '__main__':
    os.makedirs(CLEAN_DIR, exist_ok=True)

    with open(TXT_PATH) as f:
        file_names = [l.strip() for l in f if l.strip()]

    jobs = [
        (os.path.join(BASE_PATH, fname), fname[:5])
        for fname in file_names
        if os.path.exists(os.path.join(BASE_PATH, fname))
    ]

    missing = len(file_names) - len(jobs)
    if missing:
        print(f'Warning: {missing} files not found, skipping.\n')

    print(f'Preprocessing {len(jobs)} subjects with {N_JOBS} workers...\n')

    Parallel(n_jobs=N_JOBS)(
        delayed(preprocess_and_save)(fp, sid, CLEAN_DIR) for fp, sid in jobs
    )

    saved = len([f for f in os.listdir(CLEAN_DIR) if f.endswith('-epo.fif')])
    print(f'\nDone. {saved} files saved to: {CLEAN_DIR}')
