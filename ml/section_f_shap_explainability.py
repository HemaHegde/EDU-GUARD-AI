"""
EduGuard AI — Section F: SHAP Deep Explainability
==================================================
Generates MISSING SHAP plots:
  1. SHAP Dependence Plots (total_clicks, active_days, inactivity_days)
  2. SHAP Force Plot (waterfall) for individual students
  3. SHAP Interaction values

All figures saved to ieee_results/figures/
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

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
LABELS = {
    "total_clicks":             "Total Interaction Clicks",
    "avg_score":                "Mean Assessment Score",
    "active_days":              "Active Learning Days",
    "engagement_variability":   "Engagement Variability",
    "inactivity_days":          "Inactivity Duration (days)",
    "engagement_slope":         "Engagement Trend (Slope)",
    "assessment_consistency":   "Assessment Consistency (SD)",
}

# ── Load & train ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Section F — SHAP Deep Explainability")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].copy()
y  = df["psychological_risk"].copy()

X.columns = [LABELS[c] for c in X.columns]   # rename for plots

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", verbosity=0
)
print("  Training XGBoost ...")
xgb.fit(X_train, y_train)

# ── SHAP values ────────────────────────────────────────────────────────────────
print("  Computing SHAP values on test set ...")
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer(X_test)

feat_names = list(X_test.columns)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F1 — SHAP Dependence: Inactivity Duration
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating dependence plots ...")

dep_features = [
    ("Inactivity Duration (days)",  "Mean Assessment Score",       "fig_f1_shap_dep_inactivity.png"),
    ("Total Interaction Clicks",    "Active Learning Days",        "fig_f2_shap_dep_clicks.png"),
    ("Active Learning Days",        "Inactivity Duration (days)",  "fig_f3_shap_dep_activedays.png"),
]

for feat, color_feat, fname in dep_features:
    fig, ax = plt.subplots(figsize=(7, 5))
    feat_idx  = feat_names.index(feat)
    color_idx = feat_names.index(color_feat)

    x_vals    = X_test[feat].values
    shap_vals = shap_values.values[:, feat_idx]
    color_vals = X_test[color_feat].values

    sc = ax.scatter(x_vals, shap_vals, c=color_vals,
                    cmap="RdYlBu_r", alpha=0.55, s=18, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"Feature value: {color_feat}", fontsize=9)

    ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel(f"SHAP Value for {feat}", fontsize=11)
    ax.set_title(f"SHAP Dependence Plot: {feat}\n"
                 f"Interaction colored by: {color_feat}",
                 fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()
    print(f"  [SAVED] {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F4 — SHAP Waterfall for HIGH-RISK student
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating waterfall plots ...")

probs     = xgb.predict_proba(X_test)[:, 1]
high_risk_idx = int(np.argmax(probs))        # student with highest risk
low_risk_idx  = int(np.argmin(probs))        # student with lowest risk
mid_risk_idx  = int(np.argsort(abs(probs - 0.5))[0])  # closest to 0.5

for label, idx, fname in [
    ("High-Risk Student",    high_risk_idx, "fig_f4_shap_waterfall_highrisk.png"),
    ("Low-Risk Student",     low_risk_idx,  "fig_f5_shap_waterfall_lowrisk.png"),
    ("Borderline Student",   mid_risk_idx,  "fig_f6_shap_waterfall_borderline.png"),
]:
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[idx], max_display=7, show=False)
    plt.title(f"SHAP Waterfall — {label}  (Risk Score: {probs[idx]:.3f})",
              fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fname}")

    # Print feature values for case study
    print(f"\n    {label} — Feature Values:")
    for col in feat_names:
        print(f"      {col:<35}: {X_test.iloc[idx][col]:.2f}")
    print(f"      Predicted risk probability    : {probs[idx]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F7 — Mean |SHAP| Summary (extended version)
# ─────────────────────────────────────────────────────────────────────────────
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_df   = pd.DataFrame({
    "feature": feat_names,
    "mean_abs_shap": mean_shap
}).sort_values("mean_abs_shap", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(shap_df["feature"], shap_df["mean_abs_shap"],
               color="#C44E52", edgecolor="white", linewidth=0.5, height=0.6)
for bar, val in zip(bars, shap_df["mean_abs_shap"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11)
ax.set_title("SHAP Global Feature Importance — XGBoost (Extended)\n"
             "OULAD Disengagement Risk Model (Test Set n=6,519)",
             fontweight="bold", pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="lightgrey", linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_f7_shap_summary_extended.png"), dpi=300)
plt.close()
print("\n  [SAVED] fig_f7_shap_summary_extended.png")

# ── Save SHAP values ───────────────────────────────────────────────────────────
shap_df.to_csv(os.path.join(OUT_DIR, "section_f_shap_values.csv"), index=False)

print("\n  Section F complete.")
print("  Generated figures: fig_f1 to fig_f7\n")
