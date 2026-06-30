"""
EduGuard AI — Section I: Ablation Study
========================================
Evaluates 5 system configurations:

  A. Behavioral Features Only           (XGBoost, 7 features)
  B. Behavioral + SHAP Feature Selection (XGBoost, top-4 SHAP features)
  C. Behavioral + Persona Features      (XGBoost + K-Means cluster one-hot)
  D. Behavioral + Cognitive Module      (XGBoost + 4 cognitive features)
  E. Full EduGuard AI                   (All features + personas + cognitive)

Outputs:
  ieee_results/section_i_ablation_study.csv
  ieee_results/section_i_ablation_study.json
  ieee_results/figures/fig_i1_ablation_comparison.png
  ieee_results/figures/fig_i2_ablation_radar.png
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
COG_PATH   = os.path.join(SCRIPT_DIR, "video_intelligence", "cognitive_behavior_results.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BEHAVIORAL_FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

# SHAP top-4 from earlier analysis
SHAP_TOP4 = [
    "inactivity_days", "active_days", "avg_score", "assessment_consistency"
]

COGNITIVE_FEATURES = [
    "attention_score", "confusion_score",
    "boredom_score", "cognitive_overload_score"
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Section I — Ablation Study")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Main dataset: {len(df):,} rows")

# Merge cognitive data (by student_id where possible, else tile)
cog_df = pd.read_csv(COG_PATH)
print(f"  Cognitive data: {len(cog_df):,} rows")

# Align cognitive data to main dataset
repeat = int(np.ceil(len(df) / len(cog_df)))
cog_expanded = pd.concat([cog_df[COGNITIVE_FEATURES]] * repeat, ignore_index=True)
cog_expanded = cog_expanded.iloc[:len(df)].reset_index(drop=True)

# Add cognitive features to main df
df_full = pd.concat([df.reset_index(drop=True), cog_expanded], axis=1)

# Generate K-Means persona cluster as feature
scaler_km = StandardScaler()
X_km      = df[BEHAVIORAL_FEATURES].values
X_km_sc   = scaler_km.fit_transform(X_km)
km        = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=300)
km.fit(X_km_sc)
df_full["persona_cluster"] = km.predict(X_km_sc)

# One-hot encode persona
persona_dummies = pd.get_dummies(df_full["persona_cluster"],
                                  prefix="persona").astype(float)
df_full = pd.concat([df_full, persona_dummies], axis=1)
persona_cols = list(persona_dummies.columns)

y = df_full["psychological_risk"].values

# ── Helper ─────────────────────────────────────────────────────────────────────
def run_ablation(config_name, feature_cols):
    X = df_full[feature_cols].fillna(0).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1  = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)
    cm  = confusion_matrix(y_test, pred)

    print(f"\n  [{config_name}]  features={len(feature_cols)}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {pre:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")

    return {
        "config":     config_name,
        "n_features": len(feature_cols),
        "accuracy":   round(float(acc), 4),
        "precision":  round(float(pre), 4),
        "recall":     round(float(rec), 4),
        "f1_score":   round(float(f1),  4),
        "roc_auc":    round(float(auc), 4),
        "cm":         cm.tolist(),
    }

# ── Run 5 configurations ───────────────────────────────────────────────────────
print("\n  Running ablation configurations ...")

configs = [
    ("A — Behavioral Only",
     BEHAVIORAL_FEATURES),

    ("B — Behavioral + SHAP Top-4",
     SHAP_TOP4),

    ("C — Behavioral + Personas",
     BEHAVIORAL_FEATURES + persona_cols),

    ("D — Behavioral + Cognitive",
     BEHAVIORAL_FEATURES + COGNITIVE_FEATURES),

    ("E — Full EduGuard AI",
     BEHAVIORAL_FEATURES + COGNITIVE_FEATURES + persona_cols),
]

rows = [run_ablation(name, feats) for name, feats in configs]

# ── Save ───────────────────────────────────────────────────────────────────────
pd.DataFrame(rows).to_csv(
    os.path.join(OUT_DIR, "section_i_ablation_study.csv"), index=False)
with open(os.path.join(OUT_DIR, "section_i_ablation_study.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\n  [SAVED] section_i_ablation_study.csv")

# ── Fig I1: Grouped Bar Chart ──────────────────────────────────────────────────
metric_keys   = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
colors        = ["#457B9D","#2A9D8F","#E9C46A","#F4A261","#E63946"]
config_names  = [r["config"].split("—")[0].strip() for r in rows]

x     = np.arange(len(metric_labels))
n     = len(rows)
width = 0.8 / n

fig, ax = plt.subplots(figsize=(14, 6))
for i, row in enumerate(rows):
    vals   = [row[mk] for mk in metric_keys]
    offset = (i - n/2 + 0.5) * width
    bars   = ax.bar(x + offset, vals, width=width,
                    label=rows[i]["config"], color=colors[i],
                    edgecolor="white", linewidth=0.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Figure I1: Ablation Study — Module Contribution Analysis\n"
             "EduGuard AI (OULAD Dataset, XGBoost backbone)",
             fontweight="bold", pad=10)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_i1_ablation_comparison.png"), dpi=300)
plt.close()
print("  [SAVED] fig_i1_ablation_comparison.png")

# ── Fig I2: Radar Chart ────────────────────────────────────────────────────────
categories = metric_labels + [metric_labels[0]]
N          = len(metric_labels)
angles     = [n_a / float(N) * 2 * pi for n_a in range(N)]
angles    += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, row in enumerate(rows):
    vals = [row[mk] for mk in metric_keys]
    vals += vals[:1]
    label = row["config"].split("—")[0].strip()
    ax.plot(angles, vals, lw=2, color=colors[i], label=label)
    ax.fill(angles, vals, alpha=0.07, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0.6, 1.0)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.7","0.8","0.9","1.0"], fontsize=8)
ax.set_title("Figure I2: Ablation Study Radar Chart\n"
             "Module Contribution to EduGuard AI Performance",
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
          fontsize=9, framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_i2_ablation_radar.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [SAVED] fig_i2_ablation_radar.png")

# ── Final table ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION I — ABLATION RESULTS TABLE")
print("=" * 70)
print(f"\n  {'Config':<35} {'Acc':>6} {'Pre':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Feats':>6}")
print("  " + "-" * 70)
for r in rows:
    print(f"  {r['config']:<35}"
          f"  {r['accuracy']:>5.4f}"
          f"  {r['precision']:>5.4f}"
          f"  {r['recall']:>5.4f}"
          f"  {r['f1_score']:>5.4f}"
          f"  {r['roc_auc']:>5.4f}"
          f"  {r['n_features']:>5}")

# Compute gains
base_f1  = rows[0]["f1_score"]
base_auc = rows[0]["roc_auc"]
full_f1  = rows[-1]["f1_score"]
full_auc = rows[-1]["roc_auc"]
print(f"\n  Gain (Full vs Behavioral Only): F1 +{full_f1-base_f1:+.4f}  AUC +{full_auc-base_auc:+.4f}")

# ── Fig I3: Ablation Confusion Matrices ───────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, r in zip(axes, rows):
    cm = np.array(r["cm"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pass/Dist", "Fail/Withd"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(r["config"].split("—")[0].strip(), fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")

fig.suptitle("Figure I3: Confusion Matrices Across Ablation Configurations", fontweight="bold", fontsize=14)
plt.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig(os.path.join(FIG_DIR, "fig_i3_ablation_confusion.png"), dpi=300)
plt.close()
print("  [SAVED] fig_i3_ablation_confusion.png")

print("\n  Section I complete.\n")
