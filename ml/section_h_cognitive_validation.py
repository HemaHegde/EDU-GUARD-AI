"""
EduGuard AI — Section H: Cognitive Model Validation
====================================================
Computes full metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
from saved prediction CSV files:
  lstm_predictions.csv
  gru_predictions.csv
  temporal_cnn_predictions.csv

Also computes Ensemble metrics and generates comparison figures.

Outputs:
  ieee_results/section_h_cognitive_metrics.csv
  ieee_results/section_h_cognitive_metrics.json
  ieee_results/figures/fig_h1_cognitive_comparison.png
  ieee_results/figures/fig_h2_cognitive_roc.png
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(SCRIPT_DIR, "video_intelligence")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PRED_FILES = {
    "LSTM":         os.path.join(VID_DIR, "lstm_predictions.csv"),
    "GRU":          os.path.join(VID_DIR, "gru_predictions.csv"),
    "Temporal CNN": os.path.join(VID_DIR, "temporal_cnn_predictions.csv"),
}

# ── Compute metrics for each model ────────────────────────────────────────────
print("=" * 60)
print("Section H — Cognitive Model Validation")
print("=" * 60)

rows       = []
preds_dict = {}

for model_name, fpath in PRED_FILES.items():
    if not os.path.exists(fpath):
        print(f"  [WARN] File not found: {fpath}")
        continue

    df     = pd.read_csv(fpath).dropna()
    y_true = df["actual"].astype(int).values
    y_prob = df["predicted_probability"].values
    y_pred = df["predicted_class"].astype(int).values

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred)

    print(f"\n  [{model_name}]  n={len(y_true)}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {pre:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    Conf. Matrix:\n      {cm}")

    preds_dict[model_name] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}

    rows.append({
        "model":     model_name,
        "n_samples": len(y_true),
        "accuracy":  round(acc, 4),
        "precision": round(pre, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1,  4),
        "roc_auc":   round(auc, 4),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    })

# ── Ensemble: average of 3 model probabilities ────────────────────────────────
if len(preds_dict) == 3:
    names  = list(preds_dict.keys())
    y_true = preds_dict[names[0]]["y_true"]
    # align lengths (take common minimum)
    min_len = min(len(preds_dict[n]["y_prob"]) for n in names)
    ens_prob = np.mean([preds_dict[n]["y_prob"][:min_len] for n in names], axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    yt = y_true[:min_len]

    acc = accuracy_score(yt, ens_pred)
    pre = precision_score(yt, ens_pred, zero_division=0)
    rec = recall_score(yt, ens_pred, zero_division=0)
    f1  = f1_score(yt, ens_pred, zero_division=0)
    auc = roc_auc_score(yt, ens_prob)
    cm  = confusion_matrix(yt, ens_pred)

    print(f"\n  [Ensemble (LSTM+GRU+CNN average)]  n={min_len}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {pre:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")

    preds_dict["Ensemble"] = {"y_true": yt, "y_prob": ens_prob, "y_pred": ens_pred}
    rows.append({
        "model":     "Ensemble (LSTM+GRU+CNN)",
        "n_samples": min_len,
        "accuracy":  round(acc, 4),
        "precision": round(pre, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1,  4),
        "roc_auc":   round(auc, 4),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    })

# ── Save ───────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(OUT_DIR, "section_h_cognitive_metrics.csv"), index=False)
with open(os.path.join(OUT_DIR, "section_h_cognitive_metrics.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\n  [SAVED] section_h_cognitive_metrics.csv")

# ── Fig H1: Bar comparison ─────────────────────────────────────────────────────
model_names   = [r["model"] for r in rows]
metric_keys   = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
colors        = ["#457B9D","#2A9D8F","#E9C46A","#E63946"]

x     = np.arange(len(metric_labels))
n     = len(model_names)
width = 0.8 / n

fig, ax = plt.subplots(figsize=(13, 5))
for i, row in enumerate(rows):
    vals   = [row[mk] for mk in metric_keys]
    offset = (i - n/2 + 0.5) * width
    bars   = ax.bar(x + offset, vals, width=width,
                    label=row["model"], color=colors[i % len(colors)],
                    edgecolor="white", linewidth=0.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Figure H1: Cognitive Model Comparison\n"
             "LSTM vs GRU vs Temporal CNN vs Ensemble (EdNet Sequences)",
             fontweight="bold", pad=10)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_h1_cognitive_comparison.png"), dpi=300)
plt.close()
print("  [SAVED] fig_h1_cognitive_comparison.png")

# ── Fig H2: ROC curves ────────────────────────────────────────────────────────
roc_colors = ["#457B9D","#2A9D8F","#E9C46A","#E63946"]
fig, ax    = plt.subplots(figsize=(7, 6))

for i, (mname, data) in enumerate(preds_dict.items()):
    fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
    auc_val     = roc_auc_score(data["y_true"], data["y_prob"])
    ls = "-" if mname == "Ensemble" else "--"
    ax.plot(fpr, tpr, lw=2.2, color=roc_colors[i % len(roc_colors)],
            linestyle=ls, label=f"{mname}  (AUC={auc_val:.3f})")

ax.plot([0,1],[0,1],"k:", lw=1.2, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title("Figure H2: ROC Curves — Cognitive Sequence Models\n"
             "EdNet Disengagement Risk Detection",
             fontweight="bold", pad=10)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_h2_cognitive_roc.png"), dpi=300)
plt.close()
print("  [SAVED] fig_h2_cognitive_roc.png")

# ── Final table ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SECTION H — COGNITIVE MODEL RESULTS TABLE")
print("=" * 65)
print(f"\n  {'Model':<28} {'Acc':>6} {'Pre':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("  " + "-" * 65)
for r in rows:
    print(f"  {r['model']:<28}"
          f"  {r['accuracy']:>5.4f}"
          f"  {r['precision']:>5.4f}"
          f"  {r['recall']:>5.4f}"
          f"  {r['f1_score']:>5.4f}"
          f"  {r['roc_auc']:>5.4f}")
print("\n  Section H complete.\n")
