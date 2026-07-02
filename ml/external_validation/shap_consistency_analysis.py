"""
EduGuard-AI :: External Validation Pipeline
============================================
Script: shap_consistency_analysis.py

PURPOSE (strict scope)
-----------------------
Evaluate SHAP explanation consistency between the OULAD training
population and the reconstructed ASSISTments 2012-2013 external
validation population, using the SAME frozen OULAD-trained XGBoost
psychological risk model and the SAME TreeSHAP explainer.

This is an EXPLAINABILITY-CONSISTENCY ANALYSIS, not a model evaluation.
It answers "does the model rely on the same behavioural signals in a
different population?", not "is the model's prediction correct?".
Accordingly, this script deliberately does NOT:
    - retrain, fine-tune, or call .fit() on any model
    - compute, import, or reference accuracy, precision, recall, F1,
      ROC-AUC, calibration, or any other label-dependent metric
    - compute confusion matrices
    - evaluate predicted labels against ground truth of any kind
ASSISTments has no `psychological_risk` ground truth, and even for
OULAD this script is not concerned with prediction correctness -- only
with whether the model's SHAP-based feature attributions are stable
across the two populations.

Run:
    python shap_consistency_analysis.py

Paths are resolved automatically relative to this script's location, so
the script works regardless of where the repository is cloned. Only the
CONFIGURATION section below should need editing if your local file
layout differs.

Allowed dependencies: pandas, numpy, shap, matplotlib, scipy, joblib,
pathlib (standard library `logging` is used for structured console
output).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: safe for headless / CI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("shap_consistency_analysis")


# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/external_validation/shap_consistency_analysis.py
# so the project root is two directories above this file. Resolving it
# from __file__ (rather than the current working directory) means the
# script behaves identically no matter where it is invoked from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # ml/external_validation -> ml -> project_root

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Frozen, already-trained OULAD XGBoost psychological risk model. This
# model is loaded via joblib and used strictly in inference / explanation
# mode. It is never retrained and .fit() is never called on it.
MODEL_PATH = PROJECT_ROOT / "ml" / "academic_risk_xgboost.pkl"

# OULAD training population (source distribution the model was fit on).
OULAD_DATA_PATH = PROJECT_ROOT / "ml" / "final_student_psychology_dataset.csv"

# Reconstructed ASSISTments external validation population, produced by
# preprocess_assistments.py.
ASSISTMENTS_DATA_PATH = (
    PROJECT_ROOT / "datasets" / "ASSISTments" / "assistments_student_profiles.csv"
)

# Output locations.
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
REPORT_DIR = SCRIPT_DIR / "reports"

OULAD_SHAP_VALUES_PATH = OUTPUT_DIR / "oulad_shap_values.csv"
ASSISTMENTS_SHAP_VALUES_PATH = OUTPUT_DIR / "assistments_shap_values.csv"
OULAD_GLOBAL_IMPORTANCE_PATH = OUTPUT_DIR / "oulad_global_importance.csv"
ASSISTMENTS_GLOBAL_IMPORTANCE_PATH = OUTPUT_DIR / "assistments_global_importance.csv"
RANK_COMPARISON_PATH = OUTPUT_DIR / "shap_rank_comparison.csv"

OULAD_BEESWARM_PATH = FIGURE_DIR / "oulad_shap_beeswarm.png"
ASSISTMENTS_BEESWARM_PATH = FIGURE_DIR / "assistments_shap_beeswarm.png"
OULAD_BAR_PATH = FIGURE_DIR / "oulad_shap_bar.png"
ASSISTMENTS_BAR_PATH = FIGURE_DIR / "assistments_shap_bar.png"
RANK_COMPARISON_FIG_PATH = FIGURE_DIR / "shap_rank_comparison.png"

REPORT_PATH = REPORT_DIR / "shap_consistency_report.md"

# --------------------------------------------------------------------------
# FIXED FEATURE SCHEMA
# --------------------------------------------------------------------------
# These seven columns are FIXED by the original OULAD training pipeline
# (see preprocess_oulad.py) and are the exact feature set the frozen
# model expects. Using the identical feature space for both populations
# is what makes a SHAP comparison meaningful.
REQUIRED_FEATURE_COLUMNS = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
]

FEATURE_LABELS = {
    "total_clicks": "Total Interaction Clicks",
    "avg_score": "Mean Assessment Score",
    "active_days": "Active Learning Days",
    "engagement_variability": "Engagement Variability",
    "inactivity_days": "Inactivity Duration (days)",
    "engagement_slope": "Engagement Trend (Slope)",
    "assessment_consistency": "Assessment Consistency (SD)",
}

# Reasonable cap on rows used for beeswarm plotting / SHAP computation on
# very large populations, purely to keep figure rendering and runtime
# tractable. Set to None to disable subsampling.
MAX_ROWS_FOR_SHAP = 10_000
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# STEP HELPERS
# --------------------------------------------------------------------------

def load_model(path: Path):
    """Load the frozen, pre-trained OULAD XGBoost model via joblib.

    This function performs inference-mode loading ONLY. The returned
    model object is never fit, refit, or otherwise mutated anywhere in
    this script.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at: {path}\n"
            "Ensure academic_risk_xgboost.pkl has been generated and "
            "placed at the expected location."
        )
    logger.info(f"Loading frozen OULAD XGBoost model from: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    return model


def load_dataset(path: Path, name: str) -> pd.DataFrame:
    """Load a behavioural feature table from disk."""
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found at: {path}")
    logger.info(f"Loading {name} dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} {name} rows.")
    return df


def validate_feature_columns(df: pd.DataFrame, name: str) -> None:
    """Verify all seven required behavioural feature columns are present."""
    missing_columns = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The {name} dataset is missing required feature column(s): "
            f"{missing_columns}. Expected all of: {REQUIRED_FEATURE_COLUMNS}."
        )
    logger.info(f"All seven required feature columns verified present in {name} dataset.")


def prepare_feature_matrix(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Extract, order, and (if needed) subsample the fixed seven-feature
    matrix in the exact column order the frozen model was trained on.
    """
    X = df[REQUIRED_FEATURE_COLUMNS].copy()

    if MAX_ROWS_FOR_SHAP is not None and len(X) > MAX_ROWS_FOR_SHAP:
        logger.info(
            f"{name} dataset has {len(X)} rows; subsampling to "
            f"{MAX_ROWS_FOR_SHAP} rows (random_state={RANDOM_STATE}) for "
            "tractable SHAP computation."
        )
        X = X.sample(n=MAX_ROWS_FOR_SHAP, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)

    return X


def compute_shap_values(explainer: "shap.TreeExplainer", X: pd.DataFrame, name: str):
    """Compute TreeSHAP values for a feature matrix using a shared explainer."""
    logger.info(f"Computing TreeSHAP values for {name} population (n={len(X)}) ...")
    shap_values = explainer(X)
    logger.info(f"SHAP values computed for {name} population.")
    return shap_values


def save_shap_values(shap_values, X: pd.DataFrame, path: Path) -> None:
    """Persist the raw per-row, per-feature SHAP values to CSV."""
    shap_df = pd.DataFrame(shap_values.values, columns=REQUIRED_FEATURE_COLUMNS)
    shap_df.insert(0, "row_index", np.arange(len(shap_df)))
    shap_df.to_csv(path, index=False)
    logger.info(f"[SAVED] {path}")


def compute_global_importance(shap_values, name: str) -> pd.DataFrame:
    """Compute mean absolute SHAP value per feature and rank features by it."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": REQUIRED_FEATURE_COLUMNS,
        "feature_label": [FEATURE_LABELS[f] for f in REQUIRED_FEATURE_COLUMNS],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    logger.info(f"Global SHAP importance ranking computed for {name} population.")
    return importance_df


def save_global_importance(importance_df: pd.DataFrame, path: Path) -> None:
    importance_df.to_csv(path, index=False)
    logger.info(f"[SAVED] {path}")


def compute_rank_correlation(
    oulad_importance: pd.DataFrame, assistments_importance: pd.DataFrame
) -> tuple[pd.DataFrame, float, float]:
    """
    Compute Spearman rank correlation between the OULAD and ASSISTments
    feature importance rankings, aligned on feature name.
    """
    merged = oulad_importance[["feature", "feature_label", "rank", "mean_abs_shap"]].merge(
        assistments_importance[["feature", "rank", "mean_abs_shap"]],
        on="feature",
        suffixes=("_oulad", "_assistments"),
    )
    merged = merged.rename(columns={
        "rank_oulad": "oulad_rank",
        "rank_assistments": "assistments_rank",
        "mean_abs_shap_oulad": "oulad_mean_abs_shap",
        "mean_abs_shap_assistments": "assistments_mean_abs_shap",
    })
    merged = merged.sort_values("oulad_rank").reset_index(drop=True)

    rho, p_value = stats.spearmanr(merged["oulad_rank"], merged["assistments_rank"])
    logger.info(f"Spearman rank correlation (OULAD vs ASSISTments): rho={rho:.4f}, p={p_value:.4g}")
    return merged, float(rho), float(p_value)


def save_rank_comparison(merged: pd.DataFrame, rho: float, p_value: float, path: Path) -> None:
    out = merged.copy()
    out["spearman_rho"] = rho
    out["spearman_p_value"] = p_value
    out.to_csv(path, index=False)
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# FIGURE HELPERS
# --------------------------------------------------------------------------

def plot_beeswarm(shap_values, X: pd.DataFrame, title: str, path: Path) -> None:
    """Publication-quality SHAP beeswarm (summary dot) plot."""
    X_labeled = X.rename(columns=FEATURE_LABELS)
    shap_values_labeled = shap_values
    shap_values_labeled.feature_names = [FEATURE_LABELS[f] for f in X.columns]

    plt.figure(figsize=(9, 5.5))
    shap.plots.beeswarm(shap_values_labeled, max_display=len(REQUIRED_FEATURE_COLUMNS), show=False)
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_bar(importance_df: pd.DataFrame, title: str, path: Path) -> None:
    """Publication-quality mean |SHAP| bar chart, sorted ascending for barh."""
    plot_df = importance_df.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        plot_df["feature_label"], plot_df["mean_abs_shap"],
        color="#4C72B0", edgecolor="white", linewidth=0.5, height=0.6,
    )
    for bar, val in zip(bars, plot_df["mean_abs_shap"]):
        ax.text(val + 0.005 * plot_df["mean_abs_shap"].max(), bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="lightgrey", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_rank_comparison(merged: pd.DataFrame, rho: float, p_value: float, path: Path) -> None:
    """Slope chart comparing feature rank between OULAD and ASSISTments."""
    merged_sorted = merged.sort_values("oulad_rank")
    n = len(merged_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_oulad, x_assist = 0, 1

    for _, row in merged_sorted.iterrows():
        ax.plot(
            [x_oulad, x_assist],
            [row["oulad_rank"], row["assistments_rank"]],
            marker="o", markersize=7, linewidth=1.6, color="#55A868", alpha=0.85,
        )
        ax.text(x_oulad - 0.05, row["oulad_rank"], row["feature_label"],
                 ha="right", va="center", fontsize=9)
        ax.text(x_assist + 0.05, row["assistments_rank"], row["feature_label"],
                 ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(n + 0.5, 0.5)  # rank 1 at top
    ax.set_xticks([x_oulad, x_assist])
    ax.set_xticklabels(["OULAD\n(training population)", "ASSISTments\n(external population)"],
                        fontsize=10)
    ax.set_ylabel("SHAP Importance Rank (1 = most important)", fontsize=10)
    ax.set_title(
        "SHAP Feature Importance Rank Comparison\n"
        f"Spearman rho = {rho:.3f}  (p = {p_value:.3g})",
        fontweight="bold", pad=12,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------------------------

def generate_report(
    rho: float,
    p_value: float,
    merged: pd.DataFrame,
    n_oulad: int,
    n_assistments: int,
    path: Path,
) -> None:
    """Write a markdown report documenting the SHAP consistency methodology and results."""

    def interpret_rho(r: float) -> str:
        r_abs = abs(r)
        if r_abs >= 0.9:
            return "very strong"
        if r_abs >= 0.7:
            return "strong"
        if r_abs >= 0.5:
            return "moderate"
        if r_abs >= 0.3:
            return "weak"
        return "very weak / negligible"

    ranking_table_lines = ["| Feature | OULAD Rank | ASSISTments Rank | Rank Shift |",
                            "|---|---|---|---|"]
    for _, row in merged.sort_values("oulad_rank").iterrows():
        shift = int(row["assistments_rank"] - row["oulad_rank"])
        shift_str = f"+{shift}" if shift > 0 else str(shift)
        ranking_table_lines.append(
            f"| {row['feature_label']} | {int(row['oulad_rank'])} | "
            f"{int(row['assistments_rank'])} | {shift_str} |"
        )
    ranking_table = "\n".join(ranking_table_lines)

    report = f"""# SHAP Consistency Report: OULAD vs. ASSISTments

## Purpose

This report evaluates whether the **frozen, OULAD-trained XGBoost psychological
risk model** relies on the same behavioural signals when explained on the
external ASSISTments 2012-2013 population as it does on its own OULAD
training population. It is an **explainability-consistency analysis**,
not a model evaluation: it does not compute, and should not be read as
evidence for or against, predictive accuracy. ASSISTments has no
`psychological_risk` ground truth, so no label-dependent metric
(accuracy, precision, recall, F1, ROC-AUC, calibration, or confusion
matrices) is computed anywhere in this pipeline stage.

## Methodology

### Frozen model usage

A single pre-trained XGBoost classifier (`academic_risk_xgboost.pkl`) was
loaded via `joblib.load()` in inference mode only. The model was **not
retrained, fine-tuned, or refit** at any point in this analysis --
`.fit()` is never called. The exact same in-memory model object is used
to generate SHAP explanations for both populations, so any difference in
attribution patterns reflects a genuine difference in how the model
behaves on the two populations, not a difference in the model itself.

### Same explainer

A single `shap.TreeExplainer` instance was constructed once, wrapping the
frozen model, and reused to compute SHAP values for both the OULAD and
ASSISTments feature matrices. Using one explainer instance for both
populations removes explainer-configuration as a confound in any
observed differences.

### Identical feature space

Both populations are represented with the same fixed seven-feature
behavioural schema that the model was originally trained on, in the same
column order:

- Total Interaction Clicks (`total_clicks`)
- Mean Assessment Score (`avg_score`)
- Active Learning Days (`active_days`)
- Engagement Variability (`engagement_variability`)
- Inactivity Duration (`inactivity_days`)
- Engagement Trend / Slope (`engagement_slope`)
- Assessment Consistency (`assessment_consistency`)

No feature was added, dropped, renamed, or transformed differently
between the two datasets prior to SHAP computation.

### Populations

| Population | Source file | Rows explained |
|---|---|---|
| OULAD (training) | `final_student_psychology_dataset.csv` | {n_oulad} |
| ASSISTments (external) | `assistments_student_profiles.csv` | {n_assistments} |

## Feature Ranking Comparison

Features were ranked 1 (most important) through 7 (least important) in
each population by mean absolute SHAP value.

{ranking_table}

## Spearman Rank Correlation

The Spearman rank correlation coefficient between the OULAD and
ASSISTments feature importance rankings is:

**rho = {rho:.4f}** (p = {p_value:.4g})

This corresponds to a **{interpret_rho(rho)}** monotonic agreement between
the two rankings. A coefficient near +1 indicates the model draws on
behavioural features in essentially the same relative order of
importance in both populations; a coefficient near 0 or negative would
indicate that the model's reliance on individual features shifts
substantially under population change, independent of whether its
predictions on either population are correct.

## Figures

- `figures/oulad_shap_beeswarm.png` -- per-instance SHAP distribution, OULAD
- `figures/assistments_shap_beeswarm.png` -- per-instance SHAP distribution, ASSISTments
- `figures/oulad_shap_bar.png` -- mean |SHAP| global importance, OULAD
- `figures/assistments_shap_bar.png` -- mean |SHAP| global importance, ASSISTments
- `figures/shap_rank_comparison.png` -- rank slope chart, OULAD vs. ASSISTments

## Scope Note

This report and its underlying script intentionally exclude any
assessment of predictive correctness. Consistency of SHAP attributions
across populations is informative about model behaviour and feature
reliance, but is an entirely separate question from whether the model's
predicted risk labels are accurate for the ASSISTments population, which
this pipeline stage does not and cannot assess without ground-truth
labels.
"""

    path.write_text(report, encoding="utf-8")
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EduGuard-AI :: SHAP Consistency Analysis (OULAD vs. ASSISTments)")
    logger.info("=" * 70)

    # ---- 1. Load the frozen model (inference mode only) ----
    model = load_model(MODEL_PATH)

    # ---- 2. Load both datasets ----
    oulad_df = load_dataset(OULAD_DATA_PATH, "OULAD")
    assistments_df = load_dataset(ASSISTMENTS_DATA_PATH, "ASSISTments")

    # ---- 3. Validate identical feature schema ----
    validate_feature_columns(oulad_df, "OULAD")
    validate_feature_columns(assistments_df, "ASSISTments")

    X_oulad = prepare_feature_matrix(oulad_df, "OULAD")
    X_assistments = prepare_feature_matrix(assistments_df, "ASSISTments")

    # ---- 4. Build ONE shared TreeExplainer around the frozen model ----
    logger.info("Constructing shared TreeExplainer around the frozen model ...")
    explainer = shap.TreeExplainer(model)

    # ---- 5. Compute TreeSHAP values for both populations ----
    shap_values_oulad = compute_shap_values(explainer, X_oulad, "OULAD")
    shap_values_assistments = compute_shap_values(explainer, X_assistments, "ASSISTments")

    # ---- 6. Save raw SHAP values ----
    save_shap_values(shap_values_oulad, X_oulad, OULAD_SHAP_VALUES_PATH)
    save_shap_values(shap_values_assistments, X_assistments, ASSISTMENTS_SHAP_VALUES_PATH)

    # ---- 7. Compute + save global (mean |SHAP|) importance & ranking ----
    oulad_importance = compute_global_importance(shap_values_oulad, "OULAD")
    assistments_importance = compute_global_importance(shap_values_assistments, "ASSISTments")
    save_global_importance(oulad_importance, OULAD_GLOBAL_IMPORTANCE_PATH)
    save_global_importance(assistments_importance, ASSISTMENTS_GLOBAL_IMPORTANCE_PATH)

    # ---- 8. Spearman rank correlation between the two rankings ----
    merged, rho, p_value = compute_rank_correlation(oulad_importance, assistments_importance)
    save_rank_comparison(merged, rho, p_value, RANK_COMPARISON_PATH)

    # ---- 9. Figures ----
    logger.info("Generating publication-quality figures ...")
    plot_beeswarm(
        shap_values_oulad, X_oulad,
        "SHAP Summary (Beeswarm) — OULAD Training Population",
        OULAD_BEESWARM_PATH,
    )
    plot_beeswarm(
        shap_values_assistments, X_assistments,
        "SHAP Summary (Beeswarm) — ASSISTments External Population",
        ASSISTMENTS_BEESWARM_PATH,
    )
    plot_bar(
        oulad_importance,
        "SHAP Global Feature Importance — OULAD Training Population",
        OULAD_BAR_PATH,
    )
    plot_bar(
        assistments_importance,
        "SHAP Global Feature Importance — ASSISTments External Population",
        ASSISTMENTS_BAR_PATH,
    )
    plot_rank_comparison(merged, rho, p_value, RANK_COMPARISON_FIG_PATH)

    # ---- 10. Markdown report ----
    generate_report(
        rho, p_value, merged,
        n_oulad=len(X_oulad), n_assistments=len(X_assistments),
        path=REPORT_PATH,
    )

    logger.info("=" * 70)
    logger.info("SHAP consistency analysis complete.")
    logger.info(f"Spearman rank correlation (OULAD vs ASSISTments): rho={rho:.4f}, p={p_value:.4g}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
