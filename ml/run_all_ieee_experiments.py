"""
EduGuard AI — IEEE Access Master Runner
========================================
Runs ALL missing experiment scripts in sequence.
Run this file from the ml/ directory.

Usage:
    cd "c:\\Users\\hemah\\Desktop\\EDU AI\\edu-ai\\ml"
    python run_all_ieee_experiments.py

Order:
    1. Section C  — Extended Baselines (SVM, LightGBM, CatBoost)
    2. Section D  — 5-Fold Cross Validation
    3. Section E  — Statistical Significance (DeLong, McNemar)
    4. Section F  — SHAP Dependence + Waterfall Plots
    5. Section G  — Clustering Multi-Seed + PCA/t-SNE/UMAP
    6. Section H  — Cognitive Model Full Metrics (LSTM, GRU, CNN, Ensemble)
    7. Section I  — Ablation Study (5 configurations)
    8. Sections J,K,L — Fairness, Robustness, Deployment

All outputs written to:  ml/ieee_results/
All figures written to:  ml/ieee_results/figures/

Install required packages first:
    pip install lightgbm catboost umap-learn shap xgboost scikit-learn pandas numpy matplotlib seaborn

"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("Section C — Extended Baselines",          "section_c_extended_baselines.py"),
    ("Section D — 5-Fold Cross Validation",     "section_d_kfold_validation.py"),
    ("Section E — Statistical Tests",           "section_e_statistical_tests.py"),
    ("Section F — SHAP Deep Explainability",    "section_f_shap_explainability.py"),
    ("Section G — Clustering Validation",       "section_g_clustering_validation.py"),
    ("Section H — Cognitive Model Validation",  "section_h_cognitive_validation.py"),
    ("Section I — Ablation Study",              "section_i_ablation_study.py"),
    ("Sections J,K,L — Fairness/Robust/Deploy", "section_jkl_fairness_robustness_deployment.py"),
]

print("=" * 70)
print("  EduGuard AI — IEEE Access Experiment Runner")
print("  Running ALL missing sections")
print("=" * 70)
print(f"\n  Python: {sys.executable}")
print(f"  Working dir: {SCRIPT_DIR}\n")

summary = []
total_start = time.time()

for label, script_name in SCRIPTS:
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n  [SKIP] {label} — script not found: {script_name}")
        summary.append((label, "SKIPPED", 0))
        continue

    print("\n" + "=" * 70)
    print(f"  RUNNING: {label}")
    print(f"  Script : {script_name}")
    print("=" * 70 + "\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
        capture_output=False,   # show output in real-time
    )
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"ERROR (code {result.returncode})"
    summary.append((label, status, elapsed))
    print(f"\n  [{status}] {label} completed in {elapsed:.1f}s")

# ── Final Summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print("\n" + "=" * 70)
print("  MASTER RUNNER — COMPLETION SUMMARY")
print("=" * 70)
print(f"\n  {'Section':<45} {'Status':<12} {'Time':>8}")
print(f"  {'-'*68}")
for label, status, elapsed in summary:
    time_str = f"{elapsed:.0f}s" if elapsed > 0 else "—"
    print(f"  {label:<45} {status:<12} {time_str:>8}")

print(f"\n  Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

out_dir = os.path.join(SCRIPT_DIR, "ieee_results")
fig_dir = os.path.join(out_dir, "figures")
print(f"\n  All results in : {out_dir}")
print(f"  All figures in : {fig_dir}")

# ── List all output files ──────────────────────────────────────────────────────
print("\n  Output files generated:")
for fname in sorted(os.listdir(out_dir)):
    fpath = os.path.join(out_dir, fname)
    if os.path.isfile(fpath):
        sz = os.path.getsize(fpath)
        print(f"    {fname:<50} {sz/1024:>7.1f} KB")

print("\n  Figures generated:")
if os.path.exists(fig_dir):
    for fname in sorted(os.listdir(fig_dir)):
        fpath = os.path.join(fig_dir, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            print(f"    {fname:<55} {sz/1024:>7.1f} KB")

print("\n  All IEEE Access experiments complete!\n")
