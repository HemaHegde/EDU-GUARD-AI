"""
EduGuard AI - IEEE Access Evaluation Pipeline
=============================================
Recomputes ALL reported metrics from actual OULAD data:
  [OK] XGBoost  (Accuracy, Precision, Recall, F1, ROC-AUC)
  [OK] Logistic Regression metrics
  [OK] Random Forest metrics
  [OK] Confusion Matrices (all classifiers)
  [OK] SHAP mean |SHAP| values (global feature importance)
  [OK] Pearson correlations (feature vs. disengagement risk)
  [OK] One-way ANOVA (features across persona clusters)
  [OK] KMeans Silhouette Score + Davies-Bouldin Index
  [OK] Figures: ROC Curve, SHAP Summary, Confusion Matrix

Outputs saved to:  ml/ieee_results/
"""

import sys
import io
# Force UTF-8 output on Windows so Unicode symbols print cleanly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    silhouette_score, davies_bouldin_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
from xgboost import XGBClassifier
import shap
import joblib

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "academic_risk_xgboost.pkl")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR    = os.path.join(OUT_DIR, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------
# PLOT STYLE  (IEEE-clean)
# -----------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

FEATURE_COLS = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
]

FEATURE_LABELS = {
    "total_clicks":             "Total Interaction Clicks",
    "avg_score":                "Mean Assessment Score",
    "active_days":              "Active Learning Days",
    "engagement_variability":   "Engagement Variability",
    "inactivity_days":          "Inactivity Duration (days)",
    "engagement_slope":         "Engagement Trend (Slope)",
    "assessment_consistency":   "Assessment Consistency (SD)",
}

results = {}   # master dict -- serialised to JSON at the end

# =================================================================
# STEP 1 - LOAD DATA
# =================================================================

print("=" * 60)
print("STEP 1 - Loading OULAD processed dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape : {df.shape}")
print(f"  Label 0 (Not-at-risk) : {(df['psychological_risk'] == 0).sum()}")
print(f"  Label 1 (At-risk)     : {(df['psychological_risk'] == 1).sum()}")

X = df[FEATURE_COLS].copy()
y = df["psychological_risk"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"  Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

results["dataset"] = {
    "total_records" : int(len(df)),
    "label_0_count" : int((y == 0).sum()),
    "label_1_count" : int((y == 1).sum()),
    "train_size"    : int(len(X_train)),
    "test_size"     : int(len(X_test)),
    "features"      : FEATURE_COLS,
}

# =================================================================
# STEP 2 - TRAIN & EVALUATE MODELS
# =================================================================

def evaluate(name, model, X_tr, y_tr, X_te, y_te, needs_scale=False):
    """Train model and return metrics dict plus predictions."""
    if needs_scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    pre = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cm  = confusion_matrix(y_te, y_pred).tolist()
    rep = classification_report(y_te, y_pred, output_dict=True)

    print(f"\n  [{name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {pre:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    Conf.Mat  : {cm}")

    return {
        "accuracy"               : round(acc, 4),
        "precision"              : round(pre, 4),
        "recall"                 : round(rec, 4),
        "f1_score"               : round(f1,  4),
        "roc_auc"                : round(auc, 4),
        "confusion_matrix"       : cm,
        "classification_report"  : rep,
    }, y_pred, y_prob


print("\n" + "=" * 60)
print("STEP 2 - Training & Evaluating Classifiers")
print("=" * 60)

# -- Logistic Regression ------------------------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_res, lr_pred, lr_prob = evaluate(
    "Logistic Regression", lr_model,
    X_train.copy(), y_train,
    X_test.copy(),  y_test,
    needs_scale=True
)
results["logistic_regression"] = lr_res

# -- Random Forest ------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_res, rf_pred, rf_prob = evaluate(
    "Random Forest", rf_model,
    X_train, y_train,
    X_test,  y_test
)
results["random_forest"] = rf_res

# -- XGBoost ------------------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
xgb_res, xgb_pred, xgb_prob = evaluate(
    "XGBoost", xgb_model,
    X_train, y_train,
    X_test,  y_test
)
results["xgboost"] = xgb_res

# Save retrained model
joblib.dump(xgb_model, MODEL_PATH)
print(f"\n  [SAVED] XGBoost model -> {MODEL_PATH}")

# =================================================================
# STEP 3 - FIGURE 1: CONFUSION MATRICES (3-panel)
# =================================================================

print("\n" + "=" * 60)
print("STEP 3 - Figure 1: Confusion Matrices")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panel_data = [
    ("Logistic Regression", confusion_matrix(y_test, lr_pred),  "#4C72B0"),
    ("Random Forest",       confusion_matrix(y_test, rf_pred),  "#55A868"),
    ("XGBoost",             confusion_matrix(y_test, xgb_pred), "#C44E52"),
]

for ax, (title, cm_arr, color) in zip(axes, panel_data):
    sns.heatmap(
        cm_arr, annot=True, fmt="d",
        cmap=sns.light_palette(color, as_cmap=True),
        ax=ax, cbar=False,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(["Not-At-Risk (0)", "At-Risk (1)"], rotation=15)
    ax.set_yticklabels(["Not-At-Risk (0)", "At-Risk (1)"], rotation=0)

fig.suptitle(
    "Figure 1: Confusion Matrices - Disengagement Risk Classification",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
out_path = os.path.join(FIG_DIR, "fig1_confusion_matrices.png")
fig.savefig(out_path)
plt.close()
print(f"  [SAVED] Figure 1 -> fig1_confusion_matrices.png")

# =================================================================
# STEP 4 - FIGURE 2: ROC CURVES
# =================================================================

print("\n" + "=" * 60)
print("STEP 4 - Figure 2: ROC Curves")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 6))

roc_models = [
    ("Logistic Regression", lr_prob,  "#4C72B0", "--"),
    ("Random Forest",       rf_prob,  "#55A868", "-."),
    ("XGBoost",             xgb_prob, "#C44E52", "-"),
]

for label, prob, color, ls in roc_models:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, lw=2.2, color=color, linestyle=ls,
            label=f"{label}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Random Classifier (AUC = 0.500)")
ax.fill_between(
    *roc_curve(y_test, xgb_prob)[:2],
    alpha=0.08, color="#C44E52"
)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
ax.set_title(
    "Figure 2: ROC Curves - Psychological Disengagement Risk Detection",
    fontweight="bold", pad=12
)
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_roc_curves.png"))
plt.close()
print(f"  [SAVED] Figure 2 -> fig2_roc_curves.png")

# =================================================================
# STEP 4B - FIGURE 8: PRECISION-RECALL CURVE
# =================================================================

print("\n" + "=" * 60)
print("STEP 4B - Figure 8: Precision-Recall Curves")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 6))
for label, prob, color, ls in roc_models:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax.plot(rec, prec, lw=2.2, color=color, linestyle=ls,
            label=f"{label}  (AP = {ap:.3f})")

baseline = y_test.mean()
ax.plot([0, 1], [baseline, baseline], "k:", lw=1.2, label=f"Random Classifier (AP = {baseline:.3f})")
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (Positive Predictive Value)")
ax.set_title(
    "Figure 8: Precision-Recall Curves\n"
    "Disengagement Risk Detection (Imbalanced Classes)",
    fontweight="bold", pad=12
)
ax.legend(loc="lower left", framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_pr_curves.png"))
plt.close()
print(f"  [SAVED] Figure 8 -> fig8_pr_curves.png")

# =================================================================
# STEP 4C - FIGURE 9: CALIBRATION CURVE
# =================================================================

print("\n" + "=" * 60)
print("STEP 4C - Figure 9: Calibration Curves (Reliability)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

for label, prob, color, ls in roc_models:
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
    brier = brier_score_loss(y_test, prob)
    ax.plot(prob_pred, prob_true, "s-", color=color, linestyle=ls,
            label=f"{label} (Brier={brier:.3f})")

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title(
    "Figure 9: Calibration Curves (Reliability Diagram)\n"
    "Probability Estimation of Disengagement Risk",
    fontweight="bold", pad=12
)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig9_calibration_curve.png"))
plt.close()
print(f"  [SAVED] Figure 9 -> fig9_calibration_curve.png")

# =================================================================
# STEP 5 - SHAP ANALYSIS + FIGURES 3 & 4
# =================================================================

print("\n" + "=" * 60)
print("STEP 5 - SHAP Feature Importance")
print("=" * 60)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

mean_shap = np.abs(shap_values.values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature"       : FEATURE_COLS,
    "label"         : [FEATURE_LABELS[f] for f in FEATURE_COLS],
    "mean_abs_shap" : mean_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\n  Global Mean |SHAP| Values:")
for _, row in shap_df.iterrows():
    print(f"    {row['label']:<35} : {row['mean_abs_shap']:.4f}")

results["shap_importance"] = shap_df.to_dict(orient="records")

# -- Figure 3: SHAP Horizontal Bar --------------------------------
DIRECTION_COLORS = {
    "total_clicks":           "#2196F3",
    "avg_score":              "#2196F3",
    "active_days":            "#2196F3",
    "engagement_slope":       "#2196F3",
    "inactivity_days":        "#F44336",
    "engagement_variability": "#F44336",
    "assessment_consistency": "#F44336",
}

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [DIRECTION_COLORS[f] for f in shap_df["feature"]]
bars = ax.barh(
    shap_df["label"], shap_df["mean_abs_shap"],
    color=bar_colors, edgecolor="white", linewidth=0.6, height=0.62
)

for bar, val in zip(bars, shap_df["mean_abs_shap"]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9.5, color="#333333")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2196F3", label="Protective Factor (increases reduce risk)"),
    Patch(facecolor="#F44336", label="Risk Factor (increases raise risk)"),
]
ax.legend(handles=legend_elements, loc="lower right", framealpha=0.85, fontsize=9)
ax.set_xlabel("Mean Absolute SHAP Value")
ax.set_title(
    "Figure 3: SHAP Global Feature Importance\n"
    "XGBoost Disengagement Risk Model - OULAD Dataset (n = 32,593)",
    fontweight="bold", pad=10
)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="x", color="lightgrey", linestyle="--", linewidth=0.6, alpha=0.8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_shap_importance.png"))
plt.close()
print(f"  [SAVED] Figure 3 -> fig3_shap_importance.png")

# -- Figure 4: SHAP Beeswarm --------------------------------------
plt.figure(figsize=(9, 5))
shap.plots.beeswarm(shap_values, max_display=7, show=False, color_bar=True)
plt.title(
    "Figure 4: SHAP Beeswarm Plot - XGBoost Feature Impact Distribution",
    fontsize=12, fontweight="bold", pad=10
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"  [SAVED] Figure 4 -> fig4_shap_beeswarm.png")

# =================================================================
# STEP 6 - PEARSON CORRELATIONS
# =================================================================

print("\n" + "=" * 60)
print("STEP 6 - Pearson Correlation Analysis")
print("=" * 60)

correlations = []
for feat in FEATURE_COLS:
    r, p = stats.pearsonr(df[feat], df["psychological_risk"])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    correlations.append({
        "feature"     : feat,
        "label"       : FEATURE_LABELS[feat],
        "r"           : round(r, 4),
        "p_value"     : float(p),
        "p_label"     : "<0.001" if p < 0.001 else f"{p:.4f}",
        "significance": sig,
    })
    p_str = "<0.001" if p < 0.001 else f"= {p:.4f}"
    print(f"  {FEATURE_LABELS[feat]:<35}  r = {r:+.4f}  p {p_str}  {sig}")

correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
results["pearson_correlations"] = correlations

# =================================================================
# STEP 7 - ONE-WAY ANOVA
# =================================================================

print("\n" + "=" * 60)
print("STEP 7 - One-Way ANOVA (Persona Clusters)")
print("=" * 60)

df_persona   = df[df["persona"].notna()].copy()
persona_list = sorted(df_persona["persona"].unique())
n_clusters   = len(persona_list)
n_total      = len(df_persona)

print(f"  Personas : {persona_list}")
print(f"  n total  : {n_total}")

anova_results = []
for feat in FEATURE_COLS:
    groups = [
        df_persona.loc[df_persona["persona"] == lbl, feat].values
        for lbl in persona_list
    ]
    F, p = stats.f_oneway(*groups)

    grand_mean = df_persona[feat].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = ((df_persona[feat] - grand_mean) ** 2).sum()
    eta_sq     = ss_between / ss_total if ss_total != 0 else 0.0
    effect     = "Large" if eta_sq >= 0.14 else ("Medium" if eta_sq >= 0.06 else "Small")
    sig        = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    anova_results.append({
        "feature"    : feat,
        "label"      : FEATURE_LABELS[feat],
        "f_statistic": round(float(F), 2),
        "p_value"    : float(p),
        "p_label"    : "<0.001" if p < 0.001 else f"{p:.4f}",
        "significance": sig,
        "eta_squared": round(float(eta_sq), 4),
        "effect_size": effect,
    })
    print(
        f"  {FEATURE_LABELS[feat]:<35}  "
        f"F = {F:>10.2f}  eta2 = {eta_sq:.4f}  {effect}  {sig}"
    )

results["anova"] = {
    "n_clusters": n_clusters,
    "n_total"   : n_total,
    "clusters"  : persona_list,
    "results"   : anova_results,
}

# =================================================================
# STEP 8 - KMEANS VALIDATION
# =================================================================

print("\n" + "=" * 60)
print("STEP 8 - KMeans Clustering Validation")
print("=" * 60)

scaler_km  = StandardScaler()
scaled_km  = scaler_km.fit_transform(df[FEATURE_COLS])

kmeans     = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
km_labels  = kmeans.fit_predict(scaled_km)

# Silhouette on 10K sample (full 32K is slow)
rng        = np.random.RandomState(42)
sample_idx = rng.choice(len(scaled_km), size=10000, replace=False)
sil_score  = silhouette_score(scaled_km[sample_idx], km_labels[sample_idx])
db_score   = davies_bouldin_score(scaled_km, km_labels)
inertia    = kmeans.inertia_

print(f"  Silhouette Score     : {sil_score:.4f}")
print(f"  Davies-Bouldin Index : {db_score:.4f}")
print(f"  Inertia (WCSS)       : {inertia:.2f}")
sizes = pd.Series(km_labels).value_counts().sort_index().tolist()
print(f"  Cluster sizes        : {sizes}")

results["kmeans_validation"] = {
    "n_clusters"      : 6,
    "silhouette_score": round(float(sil_score), 4),
    "davies_bouldin"  : round(float(db_score),  4),
    "inertia"         : round(float(inertia),   2),
    "cluster_sizes"   : sizes,
}

# -- Figure 5: Elbow Curve ----------------------------------------
print("  Computing Elbow Curve (k=2..10) ...")
inertias = []
k_range  = range(2, 11)
for k in k_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km_tmp.fit(scaled_km)
    inertias.append(km_tmp.inertia_)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(list(k_range), inertias, "o-", color="#C44E52", lw=2.2, ms=7)
ax.axvline(x=6, color="#2196F3", linestyle="--", lw=1.6, label="Selected k = 6")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax.set_title(
    "Figure 5: KMeans Elbow Curve - Optimal Cluster Selection\n"
    "OULAD Dataset (n = 32,593), 7 Behavioral Features",
    fontweight="bold", pad=10
)
ax.legend()
ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.6, alpha=0.8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_kmeans_elbow.png"))
plt.close()
print(f"  [SAVED] Figure 5 -> fig5_kmeans_elbow.png")

# =================================================================
# STEP 9 - FIGURE 6: CORRELATION HEATMAP
# =================================================================

print("\n" + "=" * 60)
print("STEP 9 - Figure 6: Feature Correlation Heatmap")
print("=" * 60)

rename_map = {**FEATURE_LABELS, "psychological_risk": "Disengagement Risk"}
corr_data  = df[FEATURE_COLS + ["psychological_risk"]].rename(columns=rename_map).corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_data, annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    ax=ax, square=True, linewidths=0.4, linecolor="white",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.7}
)
ax.set_title(
    "Figure 6: Pearson Correlation Matrix\n"
    "Behavioral Features vs. Disengagement Risk (OULAD, n = 32,593)",
    fontweight="bold", pad=12
)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_correlation_heatmap.png"))
plt.close()
print(f"  [SAVED] Figure 6 -> fig6_correlation_heatmap.png")

# =================================================================
# STEP 10 - FIGURE 7: CLASSIFIER COMPARISON BAR CHART
# =================================================================

print("\n" + "=" * 60)
print("STEP 10 - Figure 7: Classifier Performance Comparison")
print("=" * 60)

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
lr_vals  = [lr_res["accuracy"],  lr_res["precision"],  lr_res["recall"],
            lr_res["f1_score"],  lr_res["roc_auc"]]
rf_vals  = [rf_res["accuracy"],  rf_res["precision"],  rf_res["recall"],
            rf_res["f1_score"],  rf_res["roc_auc"]]
xgb_vals = [xgb_res["accuracy"], xgb_res["precision"], xgb_res["recall"],
            xgb_res["f1_score"], xgb_res["roc_auc"]]

x     = np.arange(len(metric_names))
width = 0.24

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width,   lr_vals,  width, label="Logistic Regression", color="#4C72B0", edgecolor="white")
b2 = ax.bar(x,           rf_vals,  width, label="Random Forest",       color="#55A868", edgecolor="white")
b3 = ax.bar(x + width,   xgb_vals, width, label="XGBoost",             color="#C44E52", edgecolor="white")

for group in [b1, b2, b3]:
    for bar in group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title(
    "Figure 7: Classifier Performance Comparison\n"
    "OULAD Disengagement Risk Prediction (Test Set n = 6,519)",
    fontweight="bold", pad=10
)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig7_classifier_comparison.png"))
plt.close()
print(f"  [SAVED] Figure 7 -> fig7_classifier_comparison.png")

# =================================================================
# STEP 11 - SAVE ALL RESULTS TO JSON + CSV
# =================================================================

print("\n" + "=" * 60)
print("STEP 11 - Saving Results to JSON & CSV")
print("=" * 60)

with open(os.path.join(OUT_DIR, "all_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  [SAVED] all_metrics.json")

# Classifier summary CSV
clf_rows = [
    {"Model": "Logistic Regression", **{k: lr_res[k]  for k in ["accuracy","precision","recall","f1_score","roc_auc"]}},
    {"Model": "Random Forest",       **{k: rf_res[k]  for k in ["accuracy","precision","recall","f1_score","roc_auc"]}},
    {"Model": "XGBoost",             **{k: xgb_res[k] for k in ["accuracy","precision","recall","f1_score","roc_auc"]}},
]
pd.DataFrame(clf_rows).to_csv(os.path.join(OUT_DIR, "classifier_metrics.csv"), index=False)
print(f"  [SAVED] classifier_metrics.csv")

shap_df.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False)
print(f"  [SAVED] shap_importance.csv")

pd.DataFrame(correlations).to_csv(os.path.join(OUT_DIR, "pearson_correlations.csv"), index=False)
print(f"  [SAVED] pearson_correlations.csv")

pd.DataFrame(anova_results).to_csv(os.path.join(OUT_DIR, "anova_results.csv"), index=False)
print(f"  [SAVED] anova_results.csv")

pd.DataFrame([results["kmeans_validation"]]).to_csv(
    os.path.join(OUT_DIR, "kmeans_validation.csv"), index=False)
print(f"  [SAVED] kmeans_validation.csv")

# =================================================================
# FINAL SUMMARY
# =================================================================

print("\n" + "=" * 60)
print("  IEEE EVALUATION PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Output directory : {OUT_DIR}")

print("\n  --- Classifier Metrics ---")
print(f"  {'Model':<24} {'Acc':>6} {'Pre':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print(f"  {'-'*56}")
for row in clf_rows:
    print(
        f"  {row['Model']:<24}"
        f"  {row['accuracy']:>5.3f}"
        f"  {row['precision']:>5.3f}"
        f"  {row['recall']:>5.3f}"
        f"  {row['f1_score']:>5.3f}"
        f"  {row['roc_auc']:>5.3f}"
    )

print("\n  --- Top SHAP Features ---")
for _, row in shap_df.head(4).iterrows():
    print(f"  {row['label']:<35}  {row['mean_abs_shap']:.4f}")

print("\n  --- Clustering Validation ---")
print(f"  Silhouette Score      : {sil_score:.4f}")
print(f"  Davies-Bouldin Index  : {db_score:.4f}")
print(f"  Inertia (WCSS)        : {inertia:.2f}")

print("\n  --- Figures Generated ---")
for fname in [
    "fig1_confusion_matrices.png",
    "fig2_roc_curves.png",
    "fig3_shap_importance.png",
    "fig4_shap_beeswarm.png",
    "fig5_kmeans_elbow.png",
    "fig6_correlation_heatmap.png",
    "fig7_classifier_comparison.png",
    "fig8_pr_curves.png",
    "fig9_calibration_curve.png",
]:
    print(f"  {fname}")

print("\n  Pipeline finished successfully.\n")
