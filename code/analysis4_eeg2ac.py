#!/usr/bin/env python3
"""
Analysis 4: EEG → Attentional Control (AC) Prediction
======================================================
39 EEG features → cognitive composites and individual scores.
Same model set as Analysis 3: Ridge, Lasso, ElasticNet, XGBoost,
PCA(k) + {Ridge, Lasso, XGBoost}.

Output:
    output/analysis4_eeg2ac/
        composite_results.csv
        individual_results.csv
        correlation_fdr.csv
        eeg2ac_report.md
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.base import clone
from xgboost import XGBRegressor

from shared import (BASE_DIR, load_eeg_raw, build_39_features,
                    load_composites, load_individual_scores,
                    get_feature_cols, impute_median)

OUT_DIR = BASE_DIR / "output" / "analysis4_eeg2ac"
N_FOLDS = 10
N_PERM = 200
RS = 42
PCA_KS = [3, 5, 7, 10]

COMPOSITE_COLS = ['WMC', 'gF', 'AC', 'SuS_AC']
INDIVIDUAL_TARGETS = {
    'Antisaccade': 'AC', 'VAorient': 'AC', 'Squared_Stroop': 'AC',
    'SART': 'SuS_AC', 'SACT': 'SuS_AC', 'PVT': 'SuS_AC',
    'SymSpan': 'WMC', 'OSpan': 'WMC', 'RotSpan': 'WMC',
    'RAPM': 'gF', 'NumberSeries': 'gF', 'LetterSets': 'gF',
}


def ridge_fn():
    return RidgeCV(alphas=np.logspace(-3, 3, 50))

def lasso_fn():
    return LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=10000)

def elasticnet_fn():
    return ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9],
                        alphas=np.logspace(-3, 1, 30), cv=5, max_iter=10000)

def xgb_fn():
    return XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                        reg_alpha=0.1, reg_lambda=5, random_state=RS, verbosity=0)


MODELS = {'Ridge': ridge_fn, 'Lasso': lasso_fn,
          'ElasticNet': elasticnet_fn, 'XGBoost': xgb_fn}
PCA_MODELS = {'Ridge': ridge_fn, 'Lasso': lasso_fn, 'XGBoost': xgb_fn}


def run_cv(X, y, model_fn, kf):
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = StandardScaler(); m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr]); preds[te] = m.predict(sc.transform(X[te]))
    return preds


def run_cv_pca(X, y, k, model_fn, kf):
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = StandardScaler(); X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
        pca = PCA(n_components=k)
        m = model_fn(); m.fit(pca.fit_transform(X_tr), y[tr])
        preds[te] = m.predict(pca.transform(X_te))
    return preds


def perm_test(X, y, model_fn, real_r2, kf):
    perm_r2s = []
    for seed in range(N_PERM):
        rng = np.random.RandomState(seed); yp = rng.permutation(y)
        preds = np.full(len(y), np.nan)
        for tr, te in kf.split(X):
            sc = StandardScaler(); m = model_fn()
            m.fit(sc.fit_transform(X[tr]), yp[tr]); preds[te] = m.predict(sc.transform(X[te]))
        perm_r2s.append(r2_score(y, preds))
    return (np.sum(np.array(perm_r2s) >= real_r2) + 1) / (N_PERM + 1)


def run_all_models(X, y, kf, label):
    """Run all models (no-PCA + PCA), return list of result dicts."""
    results = []
    best_linear_r2, best_linear_fn = -999, ridge_fn
    best_nonlinear_r2 = -999

    # No-PCA models
    for mname, mfn in MODELS.items():
        preds = run_cv(X, y, mfn, kf)
        r2 = r2_score(y, preds); rho = spearmanr(y, preds).correlation
        results.append({'target': label, 'model': mname, 'r2': r2,
                        'spearman': rho, 'perm_p': np.nan, 'n': len(y)})
        if mname != 'XGBoost' and r2 > best_linear_r2:
            best_linear_r2, best_linear_fn = r2, mfn
        if mname == 'XGBoost':
            best_nonlinear_r2 = r2
        print(f"    {mname:15s} R2={r2:+.4f} rho={rho:+.4f}")

    # PCA models
    for k in PCA_KS:
        for mname, mfn in PCA_MODELS.items():
            preds = run_cv_pca(X, y, k, mfn, kf)
            r2 = r2_score(y, preds); rho = spearmanr(y, preds).correlation
            lbl = f'PCA({k})+{mname}'
            results.append({'target': label, 'model': lbl, 'r2': r2,
                            'spearman': rho, 'perm_p': np.nan, 'n': len(y)})
            print(f"    {lbl:15s} R2={r2:+.4f} rho={rho:+.4f}")

    # Perm tests: best linear + XGBoost
    preds_lin = run_cv(X, y, best_linear_fn, kf)
    r2_lin = r2_score(y, preds_lin)
    pp_lin = perm_test(X, y, best_linear_fn, r2_lin, kf)
    for r in results:
        if r['target'] == label and r['r2'] == best_linear_r2:
            r['perm_p'] = pp_lin; break

    preds_xgb = run_cv(X, y, xgb_fn, kf)
    r2_xgb = r2_score(y, preds_xgb)
    pp_xgb = perm_test(X, y, xgb_fn, r2_xgb, kf)
    for r in results:
        if r['target'] == label and r['model'] == 'XGBoost':
            r['perm_p'] = pp_xgb; break

    print(f"    Perm(linear): p={pp_lin:.3f}{'*' if pp_lin<0.05 else ''}  "
          f"Perm(XGB): p={pp_xgb:.3f}{'*' if pp_xgb<0.05 else ''}")
    return results


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eeg = build_39_features(load_eeg_raw())
    feat_cols = get_feature_cols(eeg)
    composites = load_composites()
    indiv_scores = load_individual_scores()

    merged_comp = eeg.merge(composites, on='subject', how='inner')
    X_all = impute_median(merged_comp[feat_cols].values.astype(np.float64))

    print(f"Features: {len(feat_cols)}, Subjects: {len(merged_comp)}")

    # ===== Composites =====
    print("\n--- Composite Scores ---")
    all_comp_results = []
    for target in COMPOSITE_COLS:
        y = merged_comp[target].values
        valid = np.isfinite(y)
        X, yv = X_all[valid], y[valid]
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)
        print(f"\n  {target} (N={len(yv)})")
        res = run_all_models(X, yv, kf, target)
        all_comp_results.extend(res)

    pd.DataFrame(all_comp_results).to_csv(OUT_DIR / "composite_results.csv", index=False)

    # ===== FDR correlation: EEG ↔ AC =====
    merged_ac = eeg.merge(composites[['subject', 'AC']], on='subject', how='inner')
    merged_ac = merged_ac.dropna(subset=['AC'])

    corr_rows = []
    y_ac = merged_ac['AC'].values
    for feat in feat_cols:
        x = merged_ac[feat].values
        mask = np.isfinite(x) & np.isfinite(y_ac)
        if mask.sum() < 10:
            corr_rows.append({'feature': feat, 'r': 0, 'rho': 0, 'p': 1})
            continue
        r, p = pearsonr(x[mask], y_ac[mask])
        rho = spearmanr(x[mask], y_ac[mask]).correlation
        corr_rows.append({'feature': feat, 'r': r, 'rho': rho, 'p': p})

    corr_df = pd.DataFrame(corr_rows)
    rej, qvals, _, _ = multipletests(corr_df['p'], method='fdr_bh', alpha=0.05)
    corr_df['fdr_q'] = qvals
    corr_df['fdr_significant'] = rej
    corr_df.to_csv(OUT_DIR / "correlation_fdr.csv", index=False)

    n_sig = corr_df['fdr_significant'].sum()
    print(f"\n  EEG-AC FDR-significant: {n_sig}")
    for _, row in corr_df[corr_df['fdr_significant']].sort_values('fdr_q').iterrows():
        print(f"    {row['feature']:45s} r={row['r']:+.3f} q={row['fdr_q']:.4f}")

    # ===== Individual scores (Ridge + XGBoost only, for speed) =====
    print("\n--- Individual Scores ---")
    merged_indiv = eeg.merge(
        indiv_scores[['subject'] + list(INDIVIDUAL_TARGETS.keys())],
        on='subject', how='inner')
    X_indiv = impute_median(merged_indiv[feat_cols].values.astype(np.float64))

    indiv_results = []
    for target, construct in INDIVIDUAL_TARGETS.items():
        y_raw = merged_indiv[target].values
        valid = np.isfinite(y_raw); nv = valid.sum()
        if nv < 50:
            continue
        X, yv = X_indiv[valid], y_raw[valid]
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)

        best_r2, best_rho, best_name = -999, 0, ''
        for mname, mfn in [('Ridge', ridge_fn), ('XGBoost', xgb_fn)]:
            preds = run_cv(X, yv, mfn, kf)
            r2 = r2_score(yv, preds); rho = spearmanr(yv, preds).correlation
            if r2 > best_r2:
                best_r2, best_rho, best_name = r2, rho, mname

        if best_r2 < -0.1:
            pp = 1.0
        else:
            best_fn = ridge_fn if best_name == 'Ridge' else xgb_fn
            pp = perm_test(X, yv, best_fn, best_r2, kf)

        sig = '*' if pp < 0.05 else ''
        print(f"  {target:20s} ({construct:5s}) N={nv:3d} R2={best_r2:+.4f} rho={best_rho:+.4f} p={pp:.3f}{sig}")
        indiv_results.append({'target': target, 'construct': construct,
                              'r2': best_r2, 'spearman': best_rho,
                              'perm_p': pp, 'n': nv})

    pd.DataFrame(indiv_results).to_csv(OUT_DIR / "individual_results.csv", index=False)

    # Report
    write_report(all_comp_results, indiv_results, corr_df, OUT_DIR / "eeg2ac_report.md")
    print(f"\nOutput: {OUT_DIR}")


def write_report(comp_results, indiv_results, corr_df, path):
    comp_df = pd.DataFrame(comp_results)
    lines = ["## Composite Prediction (best model per target)", ""]
    lines.append("| Target | Model | R2 | Spearman | Perm p |")
    lines.append("|--------|-------|----|----------|--------|")
    for target in COMPOSITE_COLS:
        sub = comp_df[comp_df['target'] == target].sort_values('r2', ascending=False)
        best = sub.iloc[0]
        p_str = f"{best['perm_p']:.3f}" if np.isfinite(best['perm_p']) else "-"
        lines.append(f"| {target} | {best['model']} | {best['r2']:+.4f} | {best['spearman']:+.4f} | {p_str} |")
    lines.append("")

    sig = corr_df[corr_df['fdr_significant']]
    lines.append(f"## EEG-AC FDR-significant correlations: {len(sig)}")
    lines.append("")
    if len(sig) > 0:
        lines.append("| Feature | r | q |")
        lines.append("|---------|---|---|")
        for _, row in sig.sort_values('fdr_q').iterrows():
            lines.append(f"| {row['feature']} | {row['r']:+.3f} | {row['fdr_q']:.4f} |")
    lines.append("")

    lines.append("## Individual Scores")
    lines.append("")
    lines.append("| Target | Construct | R2 | Perm p |")
    lines.append("|--------|-----------|-----|--------|")
    for r in indiv_results:
        lines.append(f"| {r['target']} | {r['construct']} | {r['r2']:+.4f} | {r['perm_p']:.3f} |")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    run()
