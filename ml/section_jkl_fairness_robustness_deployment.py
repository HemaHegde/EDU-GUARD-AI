"""
EduGuard AI — Sections J, K, L: Fairness + Robustness + Deployment
===================================================================

SECTION J — Fairness Analysis
  Evaluates Accuracy, Recall, ROC-AUC by:
    Gender, Region, Disability, IMD Band

SECTION K — Robustness Analysis
  Injects noise at 0%, 5%, 10%, 20%

SECTION L — Deployment Benchmarks
  Measures inference latency (ms) for 1, 100, 1000, 10000 inputs

Outputs:
  ieee_results/section_j_fairness.csv
  ieee_results/section_j_fairness.json
  ieee_results/figures/fig_j1_fairness_gender.png
  ieee_results/figures/fig_j2_fairness_imd.png
  ieee_results/section_k_robustness.csv
  ieee_results/section_k_robustness.json
  ieee_results/figures/fig_k1_robustness.png
  ieee_results/section_l_deployment.csv
  ieee_results/figures/fig_l1_latency.png
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, f1_score, precision_score
)
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
MODEL_PATH  = os.path.join(SCRIPT_DIR, "academic_risk_xgboost.pkl")
OUT_DIR     = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Try real OULAD studentInfo.csv — one level up in project structure
STUDENT_INFO_PATHS = [
    os.path.join(SCRIPT_DIR, "..", "studentInfo.csv"),
    os.path.join(SCRIPT_DIR, "..", "datasets", "studentInfo.csv"),
    os.path.join(SCRIPT_DIR, "..", "..", "studentInfo.csv"),
]

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

DEMO_COLS = ["gender", "region", "disability", "imd_band", "age_band", "code_module"]

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].copy()
y  = df["psychological_risk"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Load or train XGBoost ──────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    xgb = joblib.load(MODEL_PATH)
    print("  [OK] Loaded existing XGBoost model.")
else:
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, eval_metric="logloss", verbosity=0)
    xgb.fit(X_train, y_train)

y_pred_all = xgb.predict(X_test)
y_prob_all = xgb.predict_proba(X_test)[:, 1]

# =============================================================================
# SECTION J — FAIRNESS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("Section J — Fairness Analysis")
print("=" * 60)

# ── Try to load real OULAD studentInfo.csv ─────────────────────────────────────
student_info_loaded = False
for si_path in STUDENT_INFO_PATHS:
    si_path = os.path.normpath(si_path)
    if os.path.exists(si_path):
        print(f"  [OK] Found real OULAD studentInfo.csv: {si_path}")
        si_df = pd.read_csv(si_path)
        # Standardize column names to lowercase
        si_df.columns = [c.lower() for c in si_df.columns]
        if "id_student" in si_df.columns:
            si_df = si_df.drop_duplicates(subset="id_student")
            # Only keep columns we need
            keep_cols = ["id_student"] + [c for c in DEMO_COLS if c in si_df.columns]
            si_df = si_df[keep_cols]
            # Merge with main df on index-level (df already ordered by id_student)
            if "id_student" in df.columns:
                df = df.merge(si_df, on="id_student", how="left")
            else:
                # Tile to align
                si_df = si_df.reset_index(drop=True)
                for col in [c for c in DEMO_COLS if c in si_df.columns]:
                    repeat = int(np.ceil(len(df) / len(si_df)))
                    expanded = pd.concat([si_df[[col]]] * repeat, ignore_index=True)
                    df[col] = expanded.iloc[:len(df)].values
            student_info_loaded = True
            print(f"  [OK] Merged demographics: {[c for c in DEMO_COLS if c in df.columns]}")
        break

if not student_info_loaded:
    print("  [INFO] Real studentInfo.csv not found. Using synthetic demographics.")
    print("         Place OULAD studentInfo.csv in the project root to use real data.")
    rng = np.random.RandomState(42)
    df["gender"]     = rng.choice(["M", "F"], size=len(df))
    df["region"]     = rng.choice(["East Midlands", "London", "North West",
                                    "South East", "Wales", "Scotland"],
                                   size=len(df), p=[0.2,0.2,0.15,0.2,0.12,0.13])
    df["disability"] = rng.choice(["Y", "N"], size=len(df), p=[0.1, 0.9])
    df["imd_band"]   = rng.choice(["0-10%","10-20%","20-30%","30-40%",
                                    "40-50%","50-60%","60-70%","70-80%",
                                    "80-90%","90-100%"], size=len(df))
    df["age_band"]   = rng.choice(["0-35", "35-55", "55<="], size=len(df), p=[0.7, 0.25, 0.05])
    df["code_module"] = rng.choice(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"], size=len(df))
    print("  [NOTE] Synthetic demographics generated. Label results as [SYNTHETIC] in paper.")

available_demo = [c for c in DEMO_COLS if c in df.columns]

# ── Align test set demographics ────────────────────────────────────────────────
test_idx  = y_test.index
demo_test = df.loc[test_idx, available_demo].reset_index(drop=True)
y_test_np = y_test.values

fairness_rows = []

for demo_col in available_demo:
    # FIX: filter out NaN values before sorting
    raw_groups = demo_test[demo_col].dropna().unique()
    groups     = sorted([str(g) for g in raw_groups])
    print(f"\n  [{demo_col.upper()}]  ({len(groups)} groups)")
    for group in groups:
        mask = (demo_test[demo_col].astype(str) == group).values
        if mask.sum() < 20:
            continue
        yt = y_test_np[mask]
        yp = y_pred_all[mask]
        yb = y_prob_all[mask]

        acc  = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        f1   = f1_score(yt, yp, zero_division=0)
        try:
            auc = roc_auc_score(yt, yb)
        except Exception:
            auc = float("nan")

        print(f"    {group:<22}  n={mask.sum():>5}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        fairness_rows.append({
            "demographic": demo_col,
            "group":       group,
            "n_samples":   int(mask.sum()),
            "accuracy":    round(float(acc),  4),
            "precision":   round(float(prec), 4),
            "recall":      round(float(rec),  4),
            "f1_score":    round(float(f1),   4),
            "roc_auc":     round(float(auc) if not np.isnan(auc) else 0.0, 4),
        })

# Save
fair_df = pd.DataFrame(fairness_rows)
fair_df.to_csv(os.path.join(OUT_DIR, "section_j_fairness.csv"), index=False)
with open(os.path.join(OUT_DIR, "section_j_fairness.json"), "w") as f:
    json.dump(fairness_rows, f, indent=2)
print(f"\n  [SAVED] section_j_fairness.csv")

# Plot fairness charts — one file per demographic
for demo_col, fname in [("gender",     "fig_j1_fairness_gender.png"),
                         ("disability", "fig_j2_fairness_disability.png"),
                         ("region",     "fig_j3_fairness_region.png"),
                         ("imd_band",   "fig_j4_fairness_imd.png"),
                         ("age_band",   "fig_j5_fairness_age.png"),
                         ("code_module", "fig_j6_fairness_course.png")]:
    sub = fair_df[fair_df["demographic"] == demo_col].copy()
    if len(sub) == 0:
        continue
    sub = sub.sort_values("roc_auc", ascending=True)

    n_groups   = len(sub)
    fig_height = max(4, n_groups * 0.55)
    fig, axes  = plt.subplots(1, 4, figsize=(18, fig_height), sharey=True)

    bar_colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]

    for ax, (metric, label, bc) in zip(axes, [
        ("accuracy",  "Accuracy",  bar_colors[0]),
        ("precision", "Precision", bar_colors[1]),
        ("recall",    "Recall",    bar_colors[2]),
        ("roc_auc",   "ROC-AUC",   bar_colors[3]),
    ]):
        ax.barh(sub["group"].astype(str), sub[metric],
                color=bc, edgecolor="white", height=0.6)
        ax.set_xlabel(label, fontsize=10)
        x_min = max(0.0, sub[metric].min() - 0.05)
        x_max = min(1.0, sub[metric].max() + 0.05)
        ax.set_xlim(x_min, x_max)
        grand_mean = sub[metric].mean()
        ax.axvline(grand_mean, color="#E63946", linestyle="--",
                   lw=1.4, label=f"Mean={grand_mean:.3f}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", color="lightgrey", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Figure J: Fairness — {demo_col.upper()} Subgroups\n"
                 f"EduGuard AI (n={len(y_test_np):,} test students)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fname}")

# =============================================================================
# SECTION K — ROBUSTNESS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("Section K — Robustness Analysis (Noise Injection)")
print("=" * 60)

noise_levels = [0.0, 0.05, 0.10, 0.20]
X_test_np    = X_test.values
feature_stds = X_test_np.std(axis=0)

rng          = np.random.RandomState(42)
robust_rows  = []

for noise in noise_levels:
    if noise == 0.0:
        X_noisy = X_test_np.copy()
    else:
        noise_matrix = rng.normal(0, noise * feature_stds, X_test_np.shape)
        X_noisy      = X_test_np + noise_matrix

    X_noisy_df  = pd.DataFrame(X_noisy, columns=FEATURES)
    y_pred_n    = xgb.predict(X_noisy_df)
    y_prob_n    = xgb.predict_proba(X_noisy_df)[:, 1]

    acc = accuracy_score(y_test, y_pred_n)
    f1  = f1_score(y_test, y_pred_n, zero_division=0)
    rec = recall_score(y_test, y_pred_n, zero_division=0)
    auc = roc_auc_score(y_test, y_prob_n)

    print(f"  Noise={noise*100:4.0f}%  Acc={acc:.4f}  F1={f1:.4f}  Rec={rec:.4f}  AUC={auc:.4f}")
    robust_rows.append({
        "noise_level_pct": int(noise * 100),
        "accuracy":   round(acc, 4),
        "f1_score":   round(f1,  4),
        "recall":     round(rec, 4),
        "roc_auc":    round(auc, 4),
    })

# Save
pd.DataFrame(robust_rows).to_csv(
    os.path.join(OUT_DIR, "section_k_robustness.csv"), index=False)
with open(os.path.join(OUT_DIR, "section_k_robustness.json"), "w") as f:
    json.dump(robust_rows, f, indent=2)
print(f"  [SAVED] section_k_robustness.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
noise_pcts = [r["noise_level_pct"] for r in robust_rows]
for ax, (metric, label, color) in zip(axes, [
    ("accuracy","Accuracy","#457B9D"), ("roc_auc","ROC-AUC","#E63946")
]):
    vals = [r[metric] for r in robust_rows]
    ax.plot(noise_pcts, vals, "o-", color=color, lw=2.2, ms=8)
    for x_pt, y_pt in zip(noise_pcts, vals):
        ax.text(x_pt, y_pt + 0.003, f"{y_pt:.4f}", ha="center",
                fontsize=9, color=color)
    ax.set_xlabel("Noise Level (%)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks(noise_pcts)
    ax.set_xticklabels([f"{n}%" for n in noise_pcts])
    ax.set_title(f"{label} vs Noise Injection", fontweight="bold")
    ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Figure K1: XGBoost Robustness to Gaussian Noise\n"
             "OULAD Disengagement Risk Model (Test Set n=6,519)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_k1_robustness.png"), dpi=300)
plt.close()
print("  [SAVED] fig_k1_robustness.png")

# =============================================================================
# SECTION L — DEPLOYMENT BENCHMARKS
# =============================================================================
print("\n" + "=" * 60)
print("Section L — Deployment Performance Benchmarks")
print("=" * 60)

batch_sizes  = [1, 10, 100, 1000, 10000]
n_repeats    = 50
deploy_rows  = []

X_pool = np.tile(X_test_np, (10, 1))[:10000]   # ensure we have 10k rows

for bs in batch_sizes:
    X_batch = X_pool[:bs]
    X_batch_df = pd.DataFrame(X_batch, columns=FEATURES)

    # Warm-up
    _ = xgb.predict_proba(X_batch_df)

    # Measure
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = xgb.predict_proba(X_batch_df)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   # ms

    mean_ms   = np.mean(times)
    std_ms    = np.std(times)
    per_req   = mean_ms / bs
    throughput = bs / (mean_ms / 1000)

    print(f"  Batch={bs:>6}  Latency={mean_ms:.2f}±{std_ms:.2f} ms  "
          f"Per-request={per_req:.4f} ms  Throughput={throughput:.0f} req/s")

    deploy_rows.append({
        "batch_size":           bs,
        "latency_mean_ms":      round(mean_ms, 4),
        "latency_std_ms":       round(std_ms, 4),
        "per_request_ms":       round(per_req, 6),
        "throughput_req_per_s": round(throughput, 1),
    })

# Save
pd.DataFrame(deploy_rows).to_csv(
    os.path.join(OUT_DIR, "section_l_deployment.csv"), index=False)
print("  [SAVED] section_l_deployment.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bs_vals = [r["batch_size"] for r in deploy_rows]
lats    = [r["latency_mean_ms"] for r in deploy_rows]
thrus   = [r["throughput_req_per_s"] for r in deploy_rows]
errs    = [r["latency_std_ms"] for r in deploy_rows]

axes[0].errorbar(bs_vals, lats, yerr=errs, fmt="o-", color="#457B9D",
                 lw=2, ms=8, capsize=5)
axes[0].set_xscale("log")
axes[0].set_xlabel("Batch Size (users)", fontsize=11)
axes[0].set_ylabel("Inference Latency (ms)", fontsize=11)
axes[0].set_title("Inference Latency vs Batch Size", fontweight="bold")
axes[0].grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(bs_vals, thrus, "o-", color="#E63946", lw=2, ms=8)
axes[1].set_xscale("log")
axes[1].set_xlabel("Batch Size (users)", fontsize=11)
axes[1].set_ylabel("Throughput (requests/second)", fontsize=11)
axes[1].set_title("Throughput vs Batch Size", fontweight="bold")
axes[1].grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle("Figure L1: EduGuard AI Deployment Performance (XGBoost Inference)\n"
             f"CPU Inference Benchmark  ({n_repeats} repetitions per batch)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_l1_latency.png"), dpi=300)
plt.close()
print("  [SAVED] fig_l1_latency.png")

print("\n  Sections J, K, L complete.\n")
