"""
Regression prediction: ROI-based resting EEG features -> DDM parameters (a, v).

Feature selection: Use only top-N FDR-significant features from correlation results.
Features are selected PER DDM target variable (a-targets and v-targets get different feature sets).

Usage:
  python ddm_eeg_regression_topN.py              # default top_n=10
  python ddm_eeg_regression_topN.py --top_n 5
  python ddm_eeg_regression_topN.py --top_n 5 10 15 20   # sweep multiple values
  python ddm_eeg_regression_topN.py --top_n all           # use all 91 features (no selection)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import argparse
import warnings
import gc
import os

warnings.filterwarnings("ignore")

# ─── Args ───
parser = argparse.ArgumentParser()
parser.add_argument("--top_n", type=str, nargs="+", default=["10"],
                    help="Top N features: integers or 'all' for all 91 features")
raw_args = parser.parse_args()

# Parse top_n: "all" -> total number of EEG features (resolved after loading data)
_top_n_raw = raw_args.top_n

# ─── Load data ───
version = "v2"   #####manually change this
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ddm = pd.read_csv(os.path.join(base_dir, "DDM_Scores.csv"))
eeg = pd.read_csv(os.path.join(base_dir, f"preprocess_{version}.csv"))
corr = pd.read_csv(os.path.join(base_dir, "analysis", "DDM_EEG_Correlation_Results.csv"))

all_eeg_features = [c for c in eeg.columns if c != "subject"]

eeg.rename(columns={"subject": "Subject"}, inplace=True)
eeg_clean = eeg.dropna(subset=all_eeg_features).copy()

ddm["a"] = pd.to_numeric(ddm["a"], errors="coerce")
ddm["v"] = pd.to_numeric(ddm["v"], errors="coerce")
ddm_clean = ddm.dropna(subset=["a", "v"]).copy()

common_subjects = set(eeg_clean["Subject"]) & set(ddm_clean["Subject"])
print(f"Common subjects: {len(common_subjects)}")

# Resolve "all" -> total number of features
top_n_values = []
for v in _top_n_raw:
    if v.lower() == "all":
        top_n_values.append(len(all_eeg_features))
    else:
        top_n_values.append(int(v))
print(f"Top-N values: {top_n_values} (total features: {len(all_eeg_features)})")

eeg_sub = eeg_clean[eeg_clean["Subject"].isin(common_subjects)].set_index("Subject")
ddm_sub = ddm_clean[ddm_clean["Subject"].isin(common_subjects)].copy()


# ─── Feature selection from correlation results ───
def select_top_features(corr_df, ddm_col, top_n):
    """Select top N FDR-significant features for a given DDM variable.

    Strategy: pool all correlation results where DDM_Variable matches the
    target pattern (e.g., all *_a targets share features, all *_v share features),
    keep only FDR < 0.05, rank by median |Pearson_r| across methods, take top N.
    If fewer than top_n survive FDR, use all that survive + fill from nominally
    significant (p < 0.05) ranked by |r|.
    """
    # Determine if this is an 'a' or 'v' target
    if ddm_col in ("a", "v"):
        suffix = ddm_col
    else:
        suffix = ddm_col.split("_")[-1]  # Load_a -> a, Speed_Mid_v -> v

    # Pool all correlation entries for same suffix (a or v)
    a_or_v_vars = [v for v in corr_df["DDM_Variable"].unique()
                   if v == suffix or v.endswith(f"_{suffix}")]
    pooled = corr_df[corr_df["DDM_Variable"].isin(a_or_v_vars)].copy()

    # FDR-significant features
    fdr_sig = pooled[pooled["Pearson_FDR"] < 0.05].copy()

    if not fdr_sig.empty:
        # Rank by median |r| across all DDM variables of same type
        ranked = (fdr_sig.groupby("EEG_Feature")["Pearson_r"]
                  .apply(lambda x: np.median(np.abs(x)))
                  .sort_values(ascending=False))
        selected = list(ranked.head(top_n).index)
    else:
        selected = []

    # Fill if not enough
    if len(selected) < top_n:
        nom_sig = pooled[pooled["Pearson_p"] < 0.05].copy()
        ranked_nom = (nom_sig.groupby("EEG_Feature")["Pearson_r"]
                      .apply(lambda x: np.median(np.abs(x)))
                      .sort_values(ascending=False))
        for feat in ranked_nom.index:
            if feat not in selected:
                selected.append(feat)
                if len(selected) >= top_n:
                    break

    # Still not enough — fill from all by |r|
    if len(selected) < top_n:
        ranked_all = (pooled.groupby("EEG_Feature")["Pearson_r"]
                      .apply(lambda x: np.median(np.abs(x)))
                      .sort_values(ascending=False))
        for feat in ranked_all.index:
            if feat not in selected:
                selected.append(feat)
                if len(selected) >= top_n:
                    break

    return selected


# ─── DDM aggregation ───
def aggregate_overall(df):
    return df.groupby("Subject")[["a", "v"]].mean()


def aggregate_load_noload(df):
    load = df[df["Load_Condition"] == "Load"].groupby("Subject")[["a", "v"]].mean()
    load.columns = ["Load_a", "Load_v"]
    noload = df[df["Load_Condition"] == "NoLoad"].groupby("Subject")[["a", "v"]].mean()
    noload.columns = ["NoLoad_a", "NoLoad_v"]
    return load.join(noload, how="inner")


def aggregate_condition_paired_mean(df):
    conditions = ["Accuracy_Max", "Accuracy_Mid", "Neutral", "Speed_Max", "Speed_Mid"]
    result = pd.DataFrame()
    for cond in conditions:
        load_cond = df[df["Detailed_Condition"] == f"Load_{cond}"].set_index("Subject")[["a", "v"]]
        noload_cond = df[df["Detailed_Condition"] == f"NoLoad_{cond}"].set_index("Subject")[["a", "v"]]
        common_idx = load_cond.index.intersection(noload_cond.index)
        paired_mean = (load_cond.loc[common_idx] + noload_cond.loc[common_idx]) / 2
        paired_mean.columns = [f"{cond}_a", f"{cond}_v"]
        if result.empty:
            result = paired_mean
        else:
            result = result.join(paired_mean, how="inner")
    return result


# ─── Model helpers ───
ALL_MODELS = ["Ridge", "Lasso", "ElasticNet", "XGBoost", "SVR"]


def find_best_params(X, y, model_name):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if model_name == "Ridge":
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), scoring="neg_mean_absolute_error")
        m.fit(X_s, y)
        return {"alpha": m.alpha_}
    elif model_name == "Lasso":
        m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=10000)
        m.fit(X_s, y)
        return {"alpha": m.alpha_}
    elif model_name == "ElasticNet":
        m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                         alphas=np.logspace(-3, 1, 50), cv=5, max_iter=10000)
        m.fit(X_s, y)
        return {"alpha": m.alpha_, "l1_ratio": m.l1_ratio_}
    elif model_name == "XGBoost":
        param_grid = {"n_estimators": [100], "max_depth": [2, 3],
                      "learning_rate": [0.05, 0.1], "reg_alpha": [0, 0.1],
                      "reg_lambda": [1]}
        gs = GridSearchCV(
            XGBRegressor(random_state=42, verbosity=0, nthread=1),
            param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(X_s, y)
        return gs.best_params_
    elif model_name == "SVR":
        param_grid = {"C": [0.1, 1, 10], "epsilon": [0.1, 0.5],
                      "kernel": ["rbf"]}
        gs = GridSearchCV(SVR(), param_grid, cv=5,
                          scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(X_s, y)
        return gs.best_params_


def make_model(model_name, params):
    if model_name == "Ridge":
        return Ridge(alpha=params["alpha"])
    elif model_name == "Lasso":
        return Lasso(alpha=params["alpha"], max_iter=10000)
    elif model_name == "ElasticNet":
        return ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=10000)
    elif model_name == "XGBoost":
        return XGBRegressor(random_state=42, verbosity=0, nthread=1, **params)
    elif model_name == "SVR":
        return SVR(**params)


def cv_predict(X, y, model_name, params):
    """10-fold CV for all models."""
    splitter = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y))
    all_importances = []
    for train_idx, test_idx in splitter.split(X):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])
        model = make_model(model_name, params)
        model.fit(X_train_s, y[train_idx])
        y_pred[test_idx] = model.predict(X_test_s)
        if hasattr(model, "coef_"):
            all_importances.append(model.coef_)
        elif hasattr(model, "feature_importances_"):
            all_importances.append(model.feature_importances_)
        else:
            all_importances.append(np.zeros(X.shape[1]))
    return y_pred, np.mean(all_importances, axis=0)


N_PERM = 200


def permutation_test(X, y, model_name, params, actual_r2):
    """200 permutations for all models. Skip if R² <= 0."""
    if actual_r2 <= 0:
        return 1.0
    rng = np.random.RandomState(42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    n_better = 0
    for _ in range(N_PERM):
        y_perm = rng.permutation(y)
        y_pred_perm = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X[train_idx])
            X_test_s = scaler.transform(X[test_idx])
            model = make_model(model_name, params)
            model.fit(X_train_s, y_perm[train_idx])
            y_pred_perm[test_idx] = model.predict(X_test_s)
        if r2_score(y_perm, y_pred_perm) >= actual_r2:
            n_better += 1
    return (n_better + 1) / (N_PERM + 1)


def run_regression(X, y, model_name, feature_names):
    import time
    t0 = time.time()
    print(f"      [1/3] tuning...", end=" ", flush=True)
    params = find_best_params(X, y, model_name)
    print(f"[2/3] 10-fold CV...", end=" ", flush=True)
    y_pred, mean_coefs = cv_predict(X, y, model_name, params)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    if r2 <= 0:
        print(f"[3/3] perm skip (R²<=0)...", end=" ", flush=True)
    else:
        print(f"[3/3] perm test ({N_PERM}x)...", end=" ", flush=True)
    perm_p = permutation_test(X, y, model_name, params, r2)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.0f}s)")
    gc.collect()
    return {
        "R2": round(r2, 4), "MAE": round(mae, 4), "Perm_p": round(perm_p, 4),
        "Coefs": dict(zip(feature_names, np.round(mean_coefs, 4)))
    }


def run_all_models(ddm_df, eeg_df, ddm_cols, scenario_name, top_n):
    common = ddm_df.index.intersection(eeg_df.index)
    ddm_aligned = ddm_df.loc[common]

    total_models = len(ddm_cols) * len(ALL_MODELS)
    model_count = 0

    results = []
    for ddm_col in ddm_cols:
        selected_features = select_top_features(corr, ddm_col, top_n)
        eeg_aligned = eeg_df.loc[common, selected_features]
        X = eeg_aligned.values

        y = ddm_aligned[ddm_col].values
        mask = ~np.isnan(y)
        X_clean, y_clean = X[mask], y[mask]

        if len(y_clean) < 15:
            print(f"  Skipping {ddm_col}: N={len(y_clean)} too small")
            model_count += len(ALL_MODELS)
            continue

        print(f"\n  Target: {ddm_col} (N={len(y_clean)}, {len(selected_features)} features)")
        print(f"    Features: {selected_features}")

        for model_name in ALL_MODELS:
            model_count += 1
            print(f"    [{model_count}/{total_models}] {model_name}")
            res = run_regression(X_clean, y_clean, model_name, selected_features)
            sig = "***" if res["Perm_p"] < 0.001 else ("**" if res["Perm_p"] < 0.01 else ("*" if res["Perm_p"] < 0.05 else ""))
            print(f"      => R²={res['R2']:+.4f}  MAE={res['MAE']:.4f}  perm_p={res['Perm_p']:.4f}{sig}")

            row = {
                "Top_N": top_n,
                "Scenario": scenario_name,
                "Target": ddm_col,
                "Model": model_name,
                "N": len(y_clean),
                "N_Features": len(selected_features),
                "Features_Used": "; ".join(selected_features),
                "R2_CV": res["R2"],
                "MAE_CV": res["MAE"],
                "Perm_p": res["Perm_p"],
            }
            for feat, coef in res["Coefs"].items():
                row[f"Coef_{feat}"] = coef
            results.append(row)

    return pd.DataFrame(results)


# ─── Run for each top_n value ───
grand_results = []

import time as _time

for top_n in top_n_values:
    _t_start = _time.time()
    # 4 Load/NoLoad + 10 PairedMean + 2 GrandMean = 16 targets × 6 models = 96 total
    print("\n" + "=" * 80)
    print(f"TOP-N = {top_n} FDR-significant features")
    print(f"  Total: 3 scenarios, 16 targets, 5 models = 80 model runs")
    print("=" * 80)

    all_results = []

    # Load/NoLoad (4 targets × 6 models = 24)
    print(f"\n{'─' * 80}")
    print(f"  [Scenario 1/3] Load / NoLoad mean (top_n={top_n})")
    print(f"{'─' * 80}")
    ddm_ln = aggregate_load_noload(ddm_sub)
    res1 = run_all_models(ddm_ln, eeg_sub, ["Load_a", "Load_v", "NoLoad_a", "NoLoad_v"],
                          "Load/NoLoad", top_n)
    all_results.append(res1)

    # Condition-paired (10 targets × 6 models = 60)
    print(f"\n{'─' * 80}")
    print(f"  [Scenario 2/3] Condition-paired mean (top_n={top_n})")
    print(f"{'─' * 80}")
    ddm_paired = aggregate_condition_paired_mean(ddm_sub)
    res2 = run_all_models(ddm_paired, eeg_sub, list(ddm_paired.columns),
                          "PairedMean", top_n)
    all_results.append(res2)

    # Grand mean (2 targets × 6 models = 12)
    print(f"\n{'─' * 80}")
    print(f"  [Scenario 3/3] Grand mean (top_n={top_n})")
    print(f"{'─' * 80}")
    ddm_overall = aggregate_overall(ddm_sub)
    res3 = run_all_models(ddm_overall, eeg_sub, ["a", "v"], "GrandMean", top_n)
    all_results.append(res3)

    _elapsed = _time.time() - _t_start
    print(f"\n  top_n={top_n} completed in {_elapsed:.0f}s ({_elapsed/60:.1f}min)")

    combined = pd.concat(all_results, ignore_index=True)
    grand_results.append(combined)

final = pd.concat(grand_results, ignore_index=True)

# ─── Save CSV ───
output_path = os.path.join(base_dir, "analysis", "DDM_EEG_Regression_TopN_Results.csv")
final.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# ─── Summary: best per target per top_n ───
print(f"\n{'=' * 80}")
print("  SUMMARY: Best model per target (by CV R²)")
print(f"{'=' * 80}")
print(f"\n  {'Top_N':>5s}  {'Scenario':<15s} {'Target':<18s} {'Model':<12s} "
      f"{'R²':>8s} {'MAE':>8s} {'Perm_p':>8s}  {'N_Feat':>6s}")
print(f"  {'─'*5}  {'─'*15} {'─'*18} {'─'*12} {'─'*8} {'─'*8} {'─'*8}  {'─'*6}")

for top_n in top_n_values:
    tn_df = final[final["Top_N"] == top_n]
    for target in tn_df["Target"].unique():
        t_df = tn_df[tn_df["Target"] == target]
        best = t_df.loc[t_df["R2_CV"].idxmax()]
        sig = " ***" if best["Perm_p"] < 0.001 else (" **" if best["Perm_p"] < 0.01 else (" *" if best["Perm_p"] < 0.05 else ""))
        print(f"  {best['Top_N']:>5d}  {best['Scenario']:<15s} {best['Target']:<18s} "
              f"{best['Model']:<12s} {best['R2_CV']:>+8.4f} {best['MAE_CV']:>8.4f} "
              f"{best['Perm_p']:>8.4f}{sig}  {best['N_Features']:>6d}")

# ─── Comparison table if multiple top_n ───
if len(top_n_values) > 1:
    print(f"\n{'=' * 80}")
    print("  COMPARISON: Grand Mean targets across top_n values")
    print(f"{'=' * 80}")
    for target in ["a", "v"]:
        print(f"\n  Target: {target}")
        print(f"  {'Top_N':>5s}  {'Model':<12s} {'R²':>8s} {'MAE':>8s} {'Perm_p':>8s}")
        print(f"  {'─'*5}  {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
        for tn in top_n_values:
            sub = final[(final["Top_N"] == tn) & (final["Target"] == target)]
            if sub.empty:
                continue
            best = sub.loc[sub["R2_CV"].idxmax()]
            sig = " ***" if best["Perm_p"] < 0.001 else (" **" if best["Perm_p"] < 0.01 else (" *" if best["Perm_p"] < 0.05 else ""))
            print(f"  {tn:>5d}  {best['Model']:<12s} {best['R2_CV']:>+8.4f} "
                  f"{best['MAE_CV']:>8.4f} {best['Perm_p']:>8.4f}{sig}")


# ─── Generate markdown report ───
def generate_markdown_report(final):
    lines = []
    lines.append("# DDM Regression with FDR-Based Feature Selection")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("- **Feature selection**: Top-N FDR-significant features from correlation analysis")
    lines.append("- **Feature sets differ for a vs v targets** (selected separately based on correlation patterns)")
    lines.append("- **Models**: Ridge, Lasso, ElasticNet, XGBoost, SVR")
    lines.append("- **Evaluation**: 10-fold CV R², MAE, permutation test (200 iter; skipped if R² <= 0)")
    lines.append(f"- **Top-N values tested**: {sorted(final['Top_N'].unique().tolist())}")
    lines.append(f"- **N subjects**: {final['N'].max()}")
    lines.append("")

    for top_n in sorted(final["Top_N"].unique()):
        tn_df = final[final["Top_N"] == top_n]
        lines.append(f"## Top N = {top_n}")
        lines.append("")

        # Best per target
        lines.append("### Best Model per Target")
        lines.append("")
        lines.append("| Target | Model | CV R² | MAE | Perm p | Features Used |")
        lines.append("|--------|-------|--------|-----|--------|---------------|")

        for target in tn_df["Target"].unique():
            t_df = tn_df[tn_df["Target"] == target]
            best = t_df.loc[t_df["R2_CV"].idxmax()]
            sig = "***" if best["Perm_p"] < 0.001 else ("**" if best["Perm_p"] < 0.01 else ("*" if best["Perm_p"] < 0.05 else ""))
            feats = best["Features_Used"].replace("; ", ", ")
            lines.append(
                f"| {best['Target']} | {best['Model']} "
                f"| {best['R2_CV']:+.4f} | {best['MAE_CV']:.4f} "
                f"| {best['Perm_p']:.4f} {sig} | {feats} |"
            )
        lines.append("")

    # Comparison across top_n for grand mean
    if len(final["Top_N"].unique()) > 1:
        lines.append("## Comparison Across Top-N (Grand Mean)")
        lines.append("")
        for target in ["a", "v"]:
            lines.append(f"### Target: {target}")
            lines.append("")
            lines.append("| Top N | Best Model | CV R² | MAE | Perm p |")
            lines.append("|-------|-----------|--------|-----|--------|")
            for tn in sorted(final["Top_N"].unique()):
                sub = final[(final["Top_N"] == tn) & (final["Target"] == target)]
                if sub.empty:
                    continue
                best = sub.loc[sub["R2_CV"].idxmax()]
                sig = "***" if best["Perm_p"] < 0.001 else ("**" if best["Perm_p"] < 0.01 else ("*" if best["Perm_p"] < 0.05 else ""))
                lines.append(
                    f"| {tn} | {best['Model']} | {best['R2_CV']:+.4f} "
                    f"| {best['MAE_CV']:.4f} | {best['Perm_p']:.4f} {sig} |"
                )
            lines.append("")

    return "\n".join(lines)


report = generate_markdown_report(final)
report_path = os.path.join(base_dir, "analysis", f"regression_topN_results_{version}.md")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    f.write(report)
print(f"Report saved to: {report_path}")
