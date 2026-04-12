#!/usr/bin/env python3
"""
Analysis 3: EEG → Boundary Separation (a) Regression
=====================================================
39 EEG features → GrandMean a, v.

Models (no PCA):
    Ridge, Lasso, ElasticNet, XGBoost

Models (PCA):
    PCA(k) + Ridge, PCA(k) + Lasso, PCA(k) + XGBoost
    k = 3, 5, 7, 10

Permutation test (200 iter) for best linear and best nonlinear.

Output:
    output/analysis3_regression/
        regression_results.csv
        regression_report.md
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.base import clone
from xgboost import XGBRegressor

from shared import (BASE_DIR, load_eeg_raw, build_39_features,
                    load_ddm_grandmean, get_feature_cols, impute_median)

OUT_DIR = BASE_DIR / "output" / "analysis3_regression"
N_FOLDS = 10
N_PERM = 200
RS = 42


def run_cv(X, y, model_fn, kf):
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds


def run_cv_pca(X, y, k, model_fn, kf):
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        pca = PCA(n_components=k)
        X_tr_p = pca.fit_transform(X_tr)
        X_te_p = pca.transform(X_te)
        m = model_fn()
        m.fit(X_tr_p, y[tr])
        preds[te] = m.predict(X_te_p)
    return preds


def permutation_test(X, y, model_fn, real_r2, kf):
    perm_r2s = []
    for seed in range(N_PERM):
        rng = np.random.RandomState(seed)
        yp = rng.permutation(y)
        preds = np.full(len(y), np.nan)
        for tr, te in kf.split(X):
            sc = StandardScaler()
            m = model_fn()
            m.fit(sc.fit_transform(X[tr]), yp[tr])
            preds[te] = m.predict(sc.transform(X[te]))
        perm_r2s.append(r2_score(y, preds))
    return (np.sum(np.array(perm_r2s) >= real_r2) + 1) / (N_PERM + 1)


# Model factories
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


MODELS = {
    'Ridge': ridge_fn,
    'Lasso': lasso_fn,
    'ElasticNet': elasticnet_fn,
    'XGBoost': xgb_fn,
}

PCA_MODELS = {
    'Ridge': ridge_fn,
    'Lasso': lasso_fn,
    'XGBoost': xgb_fn,
}

PCA_KS = [3, 5, 7, 10]


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eeg = build_39_features(load_eeg_raw())
    ddm = load_ddm_grandmean()
    feat_cols = get_feature_cols(eeg)
    merged = eeg.merge(ddm, on='subject', how='inner').dropna(subset=['a', 'v', 't0'])

    X = impute_median(merged[feat_cols].values.astype(np.float64))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)

    results = []

    for param in ['a', 'v']:
        y = merged[param].values
        print(f"\n{'='*60}")
        print(f"GrandMean_{param} (N={len(y)})")
        print(f"{'='*60}")

        # --- No-PCA models ---
        print("\n  [No PCA]")
        best_linear_r2, best_linear_name = -999, ''
        best_nonlinear_r2, best_nonlinear_name = -999, ''

        for mname, mfn in MODELS.items():
            preds = run_cv(X, y, mfn, kf)
            r2 = r2_score(y, preds)
            rho = spearmanr(y, preds).correlation
            print(f"  {mname:15s} R²={r2:+.4f} ρ={rho:+.4f}")
            results.append({'target': f'GrandMean_{param}', 'model': mname,
                            'r2': r2, 'spearman': rho, 'perm_p': np.nan, 'n': len(y)})

            if mname == 'XGBoost':
                if r2 > best_nonlinear_r2:
                    best_nonlinear_r2, best_nonlinear_name = r2, mname
            else:
                if r2 > best_linear_r2:
                    best_linear_r2, best_linear_name = r2, mname

        # Lasso coefficients for a
        if param == 'a':
            sc = StandardScaler()
            m = lasso_fn()
            m.fit(sc.fit_transform(X), y)
            coefs = pd.Series(m.coef_, index=feat_cols)
            nz = coefs[coefs.abs() > 1e-10].sort_values(key=abs, ascending=False)
            if len(nz) > 0:
                print(f"\n  Lasso selected ({len(nz)}):")
                for feat, c in nz.head(7).items():
                    print(f"    {feat}: {c:+.4f}")

        # --- PCA models ---
        print("\n  [PCA]")
        for k in PCA_KS:
            for mname, mfn in PCA_MODELS.items():
                preds = run_cv_pca(X, y, k, mfn, kf)
                r2 = r2_score(y, preds)
                rho = spearmanr(y, preds).correlation
                label = f'PCA({k})+{mname}'
                print(f"  {label:15s} R²={r2:+.4f} ρ={rho:+.4f}")
                results.append({'target': f'GrandMean_{param}', 'model': label,
                                'r2': r2, 'spearman': rho, 'perm_p': np.nan, 'n': len(y)})

        # --- Permutation tests ---
        print("\n  [Permutation tests]")

        # Best linear (Ridge usually)
        best_lin_fn = MODELS[best_linear_name]
        preds_lin = run_cv(X, y, best_lin_fn, kf)
        r2_lin = r2_score(y, preds_lin)
        pp_lin = permutation_test(X, y, best_lin_fn, r2_lin, kf)
        print(f"  {best_linear_name}: p={pp_lin:.3f}{'*' if pp_lin<0.05 else ''}")
        for r in results:
            if r['target'] == f'GrandMean_{param}' and r['model'] == best_linear_name:
                r['perm_p'] = pp_lin

        # XGBoost
        preds_xgb = run_cv(X, y, xgb_fn, kf)
        r2_xgb = r2_score(y, preds_xgb)
        pp_xgb = permutation_test(X, y, xgb_fn, r2_xgb, kf)
        print(f"  XGBoost: p={pp_xgb:.3f}{'*' if pp_xgb<0.05 else ''}")
        for r in results:
            if r['target'] == f'GrandMean_{param}' and r['model'] == 'XGBoost':
                r['perm_p'] = pp_xgb

    reg_df = pd.DataFrame(results)
    reg_df.to_csv(OUT_DIR / "regression_results.csv", index=False)

    # Report
    with open(OUT_DIR / "regression_report.md", 'w') as f:
        for param in ['a', 'v']:
            sub = reg_df[reg_df['target'] == f'GrandMean_{param}'].sort_values('r2', ascending=False)
            f.write(f"## GrandMean_{param}\n\n")
            f.write("| Model | R² | Spearman ρ | Perm p |\n")
            f.write("|-------|----|-----------|--------|\n")
            for _, row in sub.iterrows():
                p_str = f"{row['perm_p']:.3f}" if np.isfinite(row['perm_p']) else "—"
                f.write(f"| {row['model']} | {row['r2']:+.4f} | "
                        f"{row['spearman']:+.4f} | {p_str} |\n")
            f.write("\n")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    run()
