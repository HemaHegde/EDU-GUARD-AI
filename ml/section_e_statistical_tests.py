"""
EduGuard AI — Section E: Statistical Significance Tests
========================================================
Performs:
  1. DeLong Test  — compares ROC-AUC between classifiers
  2. McNemar Test — compares prediction errors between classifiers

Comparisons:
  XGBoost vs Logistic Regression
  XGBoost vs Random Forest
  XGBoost vs LightGBM    (if installed)
  XGBoost vs CatBoost    (if installed)

Outputs:
  ieee_results/section_e_statistical_tests.csv
  ieee_results/section_e_statistical_tests.json
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

# ── DeLong AUC test (bootstrap-based) ─────────────────────────────────────────
def delong_roc_variance(ground_truth, predictions):
    """Compute variance of AUC using DeLong method."""
    order  = np.argsort(-predictions)
    label  = ground_truth[order]
    n1     = int(np.sum(label == 1))
    n0     = int(np.sum(label == 0))
    pos_scores = predictions[ground_truth == 1]
    neg_scores = predictions[ground_truth == 0]
    auc = roc_auc_score(ground_truth, predictions)
    # Wilcoxon-Mann-Whitney statistics
    v10 = np.zeros(n1)
    v01 = np.zeros(n0)
    for i, pos in enumerate(pos_scores):
        v10[i] = np.mean(pos > neg_scores) + 0.5 * np.mean(pos == neg_scores)
    for j, neg in enumerate(neg_scores):
        v01[j] = np.mean(pos_scores > neg) + 0.5 * np.mean(pos_scores == neg)
    var = (np.var(v10, ddof=1) / n1) + (np.var(v01, ddof=1) / n0)
    return auc, var

def delong_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test comparing AUC(A) vs AUC(B)."""
    auc_a, var_a = delong_roc_variance(y_true, prob_a)
    auc_b, var_b = delong_roc_variance(y_true, prob_b)
    # Covariance — assume 0 for independent models (conservative)
    se = np.sqrt(var_a + var_b)
    z  = (auc_a - auc_b) / se if se > 0 else 0.0
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p

# ── McNemar test ───────────────────────────────────────────────────────────────
def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test for paired classification comparison."""
    # b = A correct, B wrong  |  c = A wrong, B correct
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    b = np.sum(a_correct & ~b_correct)   # A right, B wrong
    c = np.sum(~a_correct & b_correct)   # A wrong, B right
    # Edwards continuity correction
    if (b + c) == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p    = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p

# ── Load & split ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Section E — Statistical Significance Tests")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].copy()
y  = df["psychological_risk"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

y_np = y_test.to_numpy()

# ── Train all models ───────────────────────────────────────────────────────────
def train_predict(model, scaled=False):
    Xtr = X_train_sc if scaled else X_train
    Xte = X_test_sc  if scaled else X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    return pred, prob

print("\n  Training all models ...")

lr_pred, lr_prob = train_predict(
    LogisticRegression(max_iter=1000, random_state=42), scaled=True)

rf_pred, rf_prob = train_predict(
    RandomForestClassifier(n_estimators=200, max_depth=10,
                           random_state=42, n_jobs=-1))

xgb_pred, xgb_prob = train_predict(
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, eval_metric="logloss", verbosity=0))

comparisons = [
    ("Logistic Regression", lr_pred, lr_prob),
    ("Random Forest",       rf_pred, rf_prob),
]

if HAS_LGB:
    lgb_pred, lgb_prob = train_predict(
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           max_depth=6, random_state=42, verbose=-1))
    comparisons.append(("LightGBM", lgb_pred, lgb_prob))

if HAS_CAT:
    cat_pred, cat_prob = train_predict(
        CatBoostClassifier(iterations=300, learning_rate=0.05,
                           depth=6, random_seed=42, verbose=0))
    comparisons.append(("CatBoost", cat_pred, cat_prob))

# ── Run tests ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STATISTICAL TEST RESULTS")
print("=" * 60)

results = []
sig_threshold = 0.05

print(f"\n  {'Comparison':<40} {'AUC_XGB':>8} {'AUC_Baseline':>12} "
      f"{'DeLong_z':>10} {'DeLong_p':>10} {'McNemar_chi2':>12} {'McNemar_p':>10} {'Sig':>4}")
print("  " + "-" * 110)

for baseline_name, b_pred, b_prob in comparisons:
    auc_xgb, auc_base, z, p_delong = delong_test(y_np, xgb_prob, b_prob)
    chi2, p_mcnemar = mcnemar_test(y_np, xgb_pred, b_pred)
    sig = "***" if p_delong < 0.001 else ("**" if p_delong < 0.01 else ("*" if p_delong < 0.05 else "ns"))

    row = {
        "comparison":        f"XGBoost vs {baseline_name}",
        "auc_xgboost":       round(auc_xgb, 4),
        "auc_baseline":      round(auc_base, 4),
        "auc_difference":    round(auc_xgb - auc_base, 4),
        "delong_z":          round(z, 4),
        "delong_p":          round(p_delong, 6),
        "mcnemar_chi2":      round(chi2, 4),
        "mcnemar_p":         round(p_mcnemar, 6),
        "significant_0.05":  bool(p_delong < sig_threshold),
        "significance_label": sig,
    }
    results.append(row)

    p_dl_str = f"{p_delong:.4f}" if p_delong >= 0.0001 else "<0.0001"
    p_mn_str = f"{p_mcnemar:.4f}" if p_mcnemar >= 0.0001 else "<0.0001"

    print(f"  {'XGBoost vs ' + baseline_name:<40} {auc_xgb:>8.4f} {auc_base:>12.4f} "
          f"{z:>10.3f} {p_dl_str:>10} {chi2:>12.3f} {p_mn_str:>10} {sig:>4}")

# ── Save ───────────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(
    os.path.join(OUT_DIR, "section_e_statistical_tests.csv"), index=False)
with open(os.path.join(OUT_DIR, "section_e_statistical_tests.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  [SAVED] section_e_statistical_tests.csv")

print("\n  Significance key:  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
print("  DeLong test: two-sided z-test comparing ROC-AUC areas")
print("  McNemar test: paired test with Edwards continuity correction")
print("\n  Section E complete.\n")
