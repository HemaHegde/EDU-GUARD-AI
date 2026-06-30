"""
EduGuard AI — Section C: Extended Baseline Classifiers
=======================================================
Trains and evaluates ALL 6 classifiers:
  1. Logistic Regression
  2. Random Forest
  3. SVM
  4. LightGBM
  5. CatBoost
  6. XGBoost (proposed)

Outputs:
  ieee_results/section_c_all_baselines.csv
  ieee_results/section_c_all_baselines.json
  ieee_results/figures/fig_c1_all_baselines_comparison.png
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Try importing optional libs ────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed. Run: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[WARN] catboost not installed. Run: pip install catboost")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Features ───────────────────────────────────────────────────────────────────
FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading dataset ...")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].copy()
y  = df["psychological_risk"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Scale once — used by LR and SVM
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Evaluate helper ────────────────────────────────────────────────────────────
def evaluate(name, model, Xtr, Xte, scaled=False):
    _Xtr = X_train_sc if scaled else Xtr
    _Xte = X_test_sc  if scaled else Xte

    model.fit(_Xtr, y_train)
    pred = model.predict(_Xte)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(_Xte)[:, 1]
    else:
        prob = model.decision_function(_Xte)
        prob = (prob - prob.min()) / (prob.max() - prob.min())

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1  = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)

    print(f"\n  [{name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {pre:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")

    return {
        "model": name,
        "accuracy":  round(float(acc), 4),
        "precision": round(float(pre), 4),
        "recall":    round(float(rec), 4),
        "f1_score":  round(float(f1),  4),
        "roc_auc":   round(float(auc), 4),
    }, pred, prob

# ── Train all classifiers ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training all 6 classifiers ...")
print("=" * 60)

rows = []
preds  = {}
probs  = {}

# 1. Logistic Regression
r, p, pb = evaluate("Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    X_train, X_test, scaled=True)
rows.append(r); preds["LR"] = p; probs["LR"] = pb

# 2. Random Forest
r, p, pb = evaluate("Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    X_train, X_test)
rows.append(r); preds["RF"] = p; probs["RF"] = pb

# 3. SVM
print("\n  [SVM] — Training (may take 1-2 min on 26k samples) ...")
r, p, pb = evaluate("SVM (RBF Kernel)",
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    X_train, X_test, scaled=True)
rows.append(r); preds["SVM"] = p; probs["SVM"] = pb

# 4. LightGBM
if HAS_LGB:
    r, p, pb = evaluate("LightGBM",
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           max_depth=6, random_state=42, verbose=-1),
        X_train, X_test)
    rows.append(r); preds["LGB"] = p; probs["LGB"] = pb
else:
    rows.append({"model": "LightGBM", "accuracy": "N/A",
                 "precision": "N/A", "recall": "N/A",
                 "f1_score": "N/A", "roc_auc": "N/A"})

# 5. CatBoost
if HAS_CAT:
    r, p, pb = evaluate("CatBoost",
        CatBoostClassifier(iterations=300, learning_rate=0.05,
                           depth=6, random_seed=42, verbose=0),
        X_train, X_test)
    rows.append(r); preds["CAT"] = p; probs["CAT"] = pb
else:
    rows.append({"model": "CatBoost", "accuracy": "N/A",
                 "precision": "N/A", "recall": "N/A",
                 "f1_score": "N/A", "roc_auc": "N/A"})

# 6. XGBoost
r, p, pb = evaluate("XGBoost (Proposed)",
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, eval_metric="logloss", verbosity=0),
    X_train, X_test)
rows.append(r); preds["XGB"] = p; probs["XGB"] = pb

# ── Save CSV ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, "section_c_all_baselines.csv")
results_df.to_csv(out_csv, index=False)
print(f"\n  [SAVED] {out_csv}")

with open(os.path.join(OUT_DIR, "section_c_all_baselines.json"), "w") as f:
    json.dump(rows, f, indent=2)

# ── Plot comparison ────────────────────────────────────────────────────────────
numeric_rows = [r for r in rows if r["accuracy"] != "N/A"]
model_names  = [r["model"] for r in numeric_rows]
metrics      = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

colors = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD"]
x = np.arange(len(metric_labels))
n = len(numeric_rows)
width = 0.8 / n

fig, ax = plt.subplots(figsize=(13, 6))
for i, row in enumerate(numeric_rows):
    vals = [row[m] for m in metrics]
    offset = (i - n/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width=width,
                  label=row["model"], color=colors[i % len(colors)],
                  edgecolor="white", linewidth=0.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Figure C1: All Classifier Performance Comparison\n"
             "OULAD Disengagement Risk Prediction (Test Set n=6,519)",
             fontweight="bold", pad=10)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_c1_all_baselines_comparison.png"), dpi=300)
plt.close()
print("  [SAVED] fig_c1_all_baselines_comparison.png")

# ── Final table ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION C — RESULTS TABLE")
print("=" * 60)
print(f"\n  {'Model':<26} {'Acc':>6} {'Pre':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print(f"  {'-'*60}")
for r in rows:
    if r["accuracy"] == "N/A":
        print(f"  {r['model']:<26}  -- Not installed --")
    else:
        print(f"  {r['model']:<26}"
              f"  {r['accuracy']:>5.3f}"
              f"  {r['precision']:>5.3f}"
              f"  {r['recall']:>5.3f}"
              f"  {r['f1_score']:>5.3f}"
              f"  {r['roc_auc']:>5.3f}")
print("\n  Section C complete.\n")
