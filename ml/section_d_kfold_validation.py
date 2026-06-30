"""
EduGuard AI — Section D: 5-Fold Cross-Validation
=================================================
Performs StratifiedKFold(n_splits=5) for ALL classifiers.

Reports:  Mean ± Standard Deviation for
  Accuracy, Precision, Recall, F1, ROC-AUC

Outputs:
  ieee_results/section_d_kfold_results.csv
  ieee_results/section_d_kfold_results.json
  ieee_results/figures/fig_d1_kfold_comparison.png
  ieee_results/figures/fig_d2_kfold_error_bars.png
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Section D — 5-Fold Cross Validation")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].values
y  = df["psychological_risk"].values
print(f"  Dataset: {len(df):,} rows  |  Features: {len(FEATURES)}")

# ── Scoring dict ───────────────────────────────────────────────────────────────
scoring = {
    "accuracy":  make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score, zero_division=0),
    "f1":        make_scorer(f1_score, zero_division=0),
    "roc_auc":   make_scorer(roc_auc_score, response_method='predict_proba'),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Define models (use Pipeline for scaled models) ─────────────────────────────
models = [
    ("Logistic Regression", Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])),
    ("Random Forest", RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ("XGBoost (Proposed)", XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0)),
]

if HAS_LGB:
    models.append(("LightGBM", lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1)))

if HAS_CAT:
    models.append(("CatBoost", CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0)))

# SVM is too slow for full 5-fold on 32k; use 10k subsample
from sklearn.utils import resample
X_svm, y_svm = resample(X, y, n_samples=10000, random_state=42, stratify=y)
models.append(("SVM (RBF, 10k sample)", Pipeline([
    ("sc", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=42))
])))

# ── Run cross-validation ───────────────────────────────────────────────────────
rows = []
metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

for name, model in models:
    print(f"\n  Running 5-fold CV: {name} ...")
    _X = X_svm if "SVM" in name else X
    _y = y_svm if "SVM" in name else y

    cv_res = cross_validate(
        model, _X, _y,
        cv=skf if "SVM" not in name else StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1 if "SVM" not in name else 1,
        return_train_score=False
    )

    row = {"model": name}
    for mk in metric_keys:
        key = f"test_{mk}"
        vals = cv_res[key]
        row[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
        row[f"{mk}_std"]  = round(float(np.std(vals)),  4)
        print(f"    {mk:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    rows.append(row)

# ── Save ───────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, "section_d_kfold_results.csv")
results_df.to_csv(out_csv, index=False)
with open(os.path.join(OUT_DIR, "section_d_kfold_results.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\n  [SAVED] section_d_kfold_results.csv")

# ── Plot 1: Error Bar Chart ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
colors = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD"]

for ax_idx, (mk, mlabel) in enumerate(zip(metric_keys, metric_labels)):
    ax = axes[ax_idx]
    names_plot = [r["model"].replace(" (Proposed)","*").replace(" (10k sample)","†") for r in rows]
    means = [r[f"{mk}_mean"] for r in rows]
    stds  = [r[f"{mk}_std"]  for r in rows]
    y_pos = np.arange(len(rows))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.barh(y_pos[i], m, xerr=s, color=colors[i % len(colors)],
                ecolor="black", capsize=4, height=0.6, align="center",
                edgecolor="white")
        ax.text(m + s + 0.002, y_pos[i], f"{m:.3f}±{s:.3f}",
                va="center", fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_plot, fontsize=8)
    ax.set_xlabel(mlabel, fontsize=9)
    ax.set_xlim(0.5, 1.05)
    ax.grid(axis="x", color="lightgrey", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Figure D1: 5-Fold Cross-Validation — Mean ± SD\n"
             "OULAD Disengagement Risk Prediction  (* = Proposed,  † = 10k subsample)",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_d1_kfold_error_bars.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [SAVED] fig_d1_kfold_error_bars.png")

# ── Print final table ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION D — 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 80)
header = f"  {'Model':<28}"
for ml in metric_labels:
    header += f"  {ml:>17}"
print(header)
print("  " + "-" * 78)
for r in rows:
    line = f"  {r['model']:<28}"
    for mk in metric_keys:
        line += f"  {r[f'{mk}_mean']:.4f} ± {r[f'{mk}_std']:.4f}"
    print(line)
print("\n  Section D complete.\n")
