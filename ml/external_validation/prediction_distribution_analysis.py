"""
EduGuard-AI :: External Validation Pipeline
============================================
Script: prediction_distribution_analysis.py

PURPOSE (strict scope)
-----------------------
Perform a purely DESCRIPTIVE analysis of the predicted-probability
distributions produced by the frozen OULAD-trained XGBoost psychological
risk model, comparing its own OULAD training population against the
reconstructed ASSISTments 2012-2013 external validation population.

This is a DISTRIBUTIONAL ANALYSIS, not a model evaluation. It answers
"what does the model's output distribution look like on each
population?", not "is the model correct?". Accordingly, this script
deliberately does NOT compute, import, or reference:
    - accuracy
    - precision / recall / F1
    - ROC-AUC
    - confusion matrices
    - calibration curves
    - any other label-dependent evaluation metric
ASSISTments has no `psychological_risk` ground truth, and even for
OULAD this script is not concerned with prediction correctness -- only
with descriptive statistics of the predicted probabilities themselves.

The "confidence" measure computed here (distance of a predicted
probability from the 0.5 decision boundary) is a purely descriptive
property of the model's OUTPUT distribution. It is explicitly NOT an
indicator of correctness and must never be interpreted as such.

Run:
    python prediction_distribution_analysis.py

Paths are resolved automatically relative to this script's location, so
the script works regardless of where the repository is cloned. Only the
CONFIGURATION section below should need editing if your local file
layout differs.

Allowed dependencies: pandas, numpy, matplotlib, scipy, joblib, pathlib
(standard library `logging` is used for structured console output).
No seaborn. No retraining. No evaluation metrics.
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
from scipy import stats

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prediction_distribution_analysis")


# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/external_validation/prediction_distribution_analysis.py
# so the project root is two directories above this file. Resolving it
# from __file__ (rather than the current working directory) ensures the
# script behaves identically no matter where it is invoked from, and
# avoids any hardcoded, OS-specific paths.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # ml/external_validation -> ml -> project_root

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Frozen, already-trained OULAD XGBoost psychological risk model. Loaded
# via joblib and used strictly in inference mode -- never retrained.
MODEL_PATH = PROJECT_ROOT / "ml" / "academic_risk_xgboost.pkl"

# OULAD training population behavioural feature table. Predicted
# probabilities for this population are generated fresh here (using the
# frozen model), since no pre-computed OULAD prediction file is part of
# this pipeline stage.
OULAD_DATA_PATH = PROJECT_ROOT / "ml" / "final_student_psychology_dataset.csv"

# Pre-computed ASSISTments external predictions, produced by
# generate_external_predictions.py.
ASSISTMENTS_PREDICTIONS_PATH = (
    PROJECT_ROOT / "ml" / "external_validation" / "outputs" / "assistments_predictions.csv"
)

# Output locations.
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
REPORT_DIR = SCRIPT_DIR / "reports"

STATISTICS_PATH = OUTPUT_DIR / "prediction_distribution_statistics.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "confidence_distribution.csv"
DECILE_PATH = OUTPUT_DIR / "predicted_probability_deciles.csv"

PROBABILITY_DIST_FIG_PATH = FIGURE_DIR / "predicted_probability_distribution.png"
DENSITY_COMPARISON_FIG_PATH = FIGURE_DIR / "prediction_density_comparison.png"
CONFIDENCE_DIST_FIG_PATH = FIGURE_DIR / "confidence_distribution.png"

REPORT_PATH = REPORT_DIR / "prediction_distribution_report.md"

# --------------------------------------------------------------------------
# FIXED FEATURE SCHEMA
# --------------------------------------------------------------------------
# These seven columns are FIXED by the original OULAD training pipeline
# (see preprocess_oulad.py). The model was trained on exactly this set,
# in this conceptual grouping; column order is enforced immediately
# before prediction to guarantee the feature matrix presented to the
# model matches what it was trained on.
REQUIRED_FEATURE_COLUMNS = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
]

# --------------------------------------------------------------------------
# RISK GROUP / CONFIDENCE BAND DEFINITIONS
# --------------------------------------------------------------------------
# Descriptive risk-probability bands. These are purely descriptive
# bucketing of the model's continuous output and are not evaluation
# thresholds of any kind.
RISK_GROUP_BINS = [0.00, 0.30, 0.70, 1.00]
RISK_GROUP_LABELS = ["Low (0.00-0.30)", "Moderate (0.30-0.70)", "High (0.70-1.00)"]

# Descriptive confidence bands over confidence = abs(p - 0.5), which
# ranges from 0.0 (maximally uncertain, p=0.5) to 0.5 (maximally
# decisive, p=0.0 or p=1.0). These bands describe how decisive the
# model's output is -- they say nothing about whether that output is
# correct.
CONFIDENCE_BINS = [0.00, 0.15, 0.35, 0.50 + 1e-9]
CONFIDENCE_LABELS = ["Low confidence (0.00-0.15)", "Medium confidence (0.15-0.35)", "High confidence (0.35-0.50)"]

# Fine-grained decile bins over the raw predicted probability itself
# (0.00-0.10 through 0.90-1.00). This gives a finer-grained descriptive
# view of the output distribution than the three-band risk grouping
# above, and is convenient for reporting tables. Purely descriptive --
# not an evaluation of correctness.
DECILE_BIN_EDGES = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.10)]
DECILE_LABELS = [
    f"{DECILE_BIN_EDGES[i]:.2f}-{DECILE_BIN_EDGES[i + 1]:.2f}"
    for i in range(len(DECILE_BIN_EDGES) - 1)
]


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


def load_oulad_dataset(path: Path) -> pd.DataFrame:
    """Load the OULAD training population behavioural feature table."""
    if not path.exists():
        raise FileNotFoundError(f"OULAD dataset not found at: {path}")
    logger.info(f"Loading OULAD dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} OULAD rows.")
    return df


def load_assistments_predictions(path: Path) -> pd.DataFrame:
    """Load the pre-computed ASSISTments external prediction table."""
    if not path.exists():
        raise FileNotFoundError(
            f"ASSISTments predictions file not found at: {path}\n"
            "Run generate_external_predictions.py first to generate this file."
        )
    logger.info(f"Loading ASSISTments predictions from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} ASSISTments prediction rows.")
    if "predicted_probability" not in df.columns:
        raise ValueError(
            f"Expected column 'predicted_probability' not found in: {path}"
        )
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


def generate_oulad_predictions(model, df: pd.DataFrame) -> pd.Series:
    """
    Apply the frozen model to the OULAD feature columns and return the
    predicted probability of class 1 (psychological_risk = 1).

    Only prediction is performed here -- no metric requiring ground
    truth is computed, by design.
    """
    X = df[REQUIRED_FEATURE_COLUMNS]
    logger.info("Generating predicted probabilities for OULAD using the frozen model ...")
    probabilities = model.predict_proba(X)[:, 1]
    logger.info(f"OULAD predicted probabilities generated for {len(probabilities)} rows.")
    return pd.Series(probabilities, name="predicted_probability")


def compute_descriptive_statistics(probs: pd.Series, name: str) -> dict:
    """
    Compute descriptive statistics of a predicted-probability series:
    mean, median, min, max, standard deviation, quartiles, and IQR.

    Purely descriptive -- no label-dependent quantity is touched.
    """
    q1, q2, q3 = np.percentile(probs, [25, 50, 75])
    stats_dict = {
        "dataset": name,
        "population": name,  # retained alongside `dataset` for backward compatibility
        "n": len(probs),
        "mean": float(probs.mean()),
        "median": float(q2),
        "minimum": float(probs.min()),
        "maximum": float(probs.max()),
        "std_dev": float(probs.std(ddof=0)),
        "q1_25th_percentile": float(q1),
        "q3_75th_percentile": float(q3),
        "iqr": float(q3 - q1),
    }
    logger.info(
        f"{name} predicted-probability stats: mean={stats_dict['mean']:.4f}, "
        f"median={stats_dict['median']:.4f}, std={stats_dict['std_dev']:.4f}, "
        f"IQR={stats_dict['iqr']:.4f}"
    )
    return stats_dict


def assign_risk_groups(probs: pd.Series) -> pd.Series:
    """Bucket predicted probabilities into descriptive Low/Moderate/High risk bands."""
    return pd.cut(
        probs, bins=RISK_GROUP_BINS, labels=RISK_GROUP_LABELS,
        include_lowest=True, right=True,
    )


def summarize_risk_groups(probs: pd.Series, name: str) -> pd.DataFrame:
    """Compute count and percentage of students in each descriptive risk group."""
    groups = assign_risk_groups(probs)
    counts = groups.value_counts().reindex(RISK_GROUP_LABELS, fill_value=0)
    percentages = (counts / len(probs) * 100.0).round(2)
    summary = pd.DataFrame({
        "population": name,
        "risk_group": RISK_GROUP_LABELS,
        "n_students": counts.values,
        "pct_students": percentages.values,
    })
    logger.info(f"{name} risk group distribution computed.")
    return summary


def compute_confidence(probs: pd.Series) -> pd.Series:
    """
    Compute a descriptive confidence measure as the distance of each
    predicted probability from the 0.5 decision boundary:

        confidence = abs(predicted_probability - 0.5)

    This measures how DECISIVE the model's output is, not whether the
    output is CORRECT. Confidence and correctness are never conflated
    anywhere in this script.
    """
    return (probs - 0.5).abs().rename("confidence")


def assign_confidence_bands(confidence: pd.Series) -> pd.Series:
    """Bucket the descriptive confidence measure into Low/Medium/High bands."""
    return pd.cut(
        confidence, bins=CONFIDENCE_BINS, labels=CONFIDENCE_LABELS,
        include_lowest=True, right=True,
    )


def summarize_confidence(confidence: pd.Series, name: str) -> pd.DataFrame:
    """Compute count and percentage of students in each descriptive confidence band."""
    bands = assign_confidence_bands(confidence)
    counts = bands.value_counts().reindex(CONFIDENCE_LABELS, fill_value=0)
    percentages = (counts / len(confidence) * 100.0).round(2)
    summary = pd.DataFrame({
        "dataset": name,
        "population": name,  # retained alongside `dataset` for backward compatibility
        "confidence_band": CONFIDENCE_LABELS,
        "n_students": counts.values,
        "pct_students": percentages.values,
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(confidence.median()),
    })
    logger.info(f"{name} confidence band distribution computed.")
    return summary


def compute_decile_distribution(probs: pd.Series, name: str) -> pd.DataFrame:
    """
    Compute the number and percentage of predicted probabilities falling
    into each 0.10-wide decile bin, from 0.00-0.10 through 0.90-1.00.

    This is a finer-grained descriptive breakdown of the raw predicted
    probability than the three-band risk grouping above, intended to
    support richer reporting tables (e.g. for a paper). It is purely
    descriptive and is not an evaluation of correctness.
    """
    bins = pd.cut(
        probs, bins=DECILE_BIN_EDGES, labels=DECILE_LABELS,
        include_lowest=True, right=True,
    )
    counts = bins.value_counts().reindex(DECILE_LABELS, fill_value=0)
    percentages = (counts / len(probs) * 100.0).round(2)
    summary = pd.DataFrame({
        "dataset": name,
        "decile_bin": DECILE_LABELS,
        "n_students": counts.values,
        "pct_students": percentages.values,
    })
    logger.info(f"{name} decile distribution computed.")
    return summary


# --------------------------------------------------------------------------
# FIGURE HELPERS
# --------------------------------------------------------------------------

def plot_probability_distribution(
    oulad_probs: pd.Series, assistments_probs: pd.Series, path: Path
) -> None:
    """Publication-quality side-by-side histograms of predicted probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, probs, label, color in [
        (axes[0], oulad_probs, "OULAD (training population)", "#4C72B0"),
        (axes[1], assistments_probs, "ASSISTments (external population)", "#DD8452"),
    ]:
        ax.hist(probs, bins=30, range=(0, 1), color=color, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.axvline(0.30, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(0.70, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Number of Students", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Predicted Probability Distribution — Frozen OULAD XGBoost Model",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_density_comparison(
    oulad_probs: pd.Series, assistments_probs: pd.Series, path: Path
) -> None:
    """Publication-quality overlaid kernel-density comparison of predicted probabilities."""
    grid = np.linspace(0.0, 1.0, 500)

    oulad_kde = stats.gaussian_kde(oulad_probs)
    assistments_kde = stats.gaussian_kde(assistments_probs)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grid, oulad_kde(grid), color="#4C72B0", linewidth=2.0, label="OULAD (training population)")
    ax.fill_between(grid, oulad_kde(grid), color="#4C72B0", alpha=0.20)
    ax.plot(grid, assistments_kde(grid), color="#DD8452", linewidth=2.0, label="ASSISTments (external population)")
    ax.fill_between(grid, assistments_kde(grid), color="#DD8452", alpha=0.20)

    ax.axvline(0.30, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.70, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Predicted Probability Density Comparison\nOULAD vs. ASSISTments (Frozen Model Output)",
        fontweight="bold", pad=12,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_confidence_distribution(
    oulad_confidence: pd.Series, assistments_confidence: pd.Series, path: Path
) -> None:
    """Publication-quality side-by-side histograms of the descriptive confidence measure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, conf, label, color in [
        (axes[0], oulad_confidence, "OULAD (training population)", "#55A868"),
        (axes[1], assistments_confidence, "ASSISTments (external population)", "#C44E52"),
    ]:
        ax.hist(conf, bins=25, range=(0, 0.5), color=color, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_xlabel("Confidence = |predicted probability - 0.5|", fontsize=10)
        ax.set_ylabel("Number of Students", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Descriptive Confidence Distribution — Frozen OULAD XGBoost Model\n"
        "(Distance from decision boundary; not a measure of correctness)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------------------------

def generate_report(
    oulad_stats: dict,
    assistments_stats: dict,
    oulad_risk_summary: pd.DataFrame,
    assistments_risk_summary: pd.DataFrame,
    oulad_confidence_summary: pd.DataFrame,
    assistments_confidence_summary: pd.DataFrame,
    oulad_decile_summary: pd.DataFrame,
    assistments_decile_summary: pd.DataFrame,
    path: Path,
) -> None:
    """Write a markdown report documenting the descriptive prediction-distribution analysis."""

    def stats_table_row(s: dict) -> str:
        return (
            f"| {s['population']} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
            f"{s['minimum']:.4f} | {s['maximum']:.4f} | {s['std_dev']:.4f} | "
            f"{s['q1_25th_percentile']:.4f} | {s['q3_75th_percentile']:.4f} | {s['iqr']:.4f} |"
        )

    stats_table = (
        "| Population | n | Mean | Median | Min | Max | Std Dev | Q1 (25th pct) | Q3 (75th pct) | IQR |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        f"{stats_table_row(oulad_stats)}\n"
        f"{stats_table_row(assistments_stats)}"
    )

    def risk_table(df: pd.DataFrame) -> str:
        lines = ["| Risk Group | n Students | % Students |", "|---|---|---|"]
        for _, row in df.iterrows():
            lines.append(f"| {row['risk_group']} | {row['n_students']} | {row['pct_students']:.2f}% |")
        return "\n".join(lines)

    def confidence_table(df: pd.DataFrame) -> str:
        lines = ["| Confidence Band | n Students | % Students |", "|---|---|---|"]
        for _, row in df.iterrows():
            lines.append(f"| {row['confidence_band']} | {row['n_students']} | {row['pct_students']:.2f}% |")
        return "\n".join(lines)

    def decile_table(oulad_df: pd.DataFrame, assistments_df: pd.DataFrame) -> str:
        lines = [
            "| Probability Decile | OULAD n | OULAD % | ASSISTments n | ASSISTments % |",
            "|---|---|---|---|---|",
        ]
        for oulad_row, assist_row in zip(oulad_df.itertuples(), assistments_df.itertuples()):
            lines.append(
                f"| {oulad_row.decile_bin} | {oulad_row.n_students} | {oulad_row.pct_students:.2f}% | "
                f"{assist_row.n_students} | {assist_row.pct_students:.2f}% |"
            )
        return "\n".join(lines)

    report = f"""# Prediction Distribution Report: OULAD vs. ASSISTments

## Purpose

This report describes the distribution of predicted probabilities
produced by the **frozen, OULAD-trained XGBoost psychological risk
model** on its own OULAD training population and on the external
ASSISTments 2012-2013 population. It is a **descriptive distributional
analysis**, not a model evaluation, and it does not compute or imply any
label-dependent performance metric. ASSISTments has no
`psychological_risk` ground truth, so accuracy, precision, recall, F1,
ROC-AUC, confusion matrices, and calibration curves are all
out of scope for this pipeline stage and are not computed here.

## Methodology

- The frozen model (`academic_risk_xgboost.pkl`) was loaded via
  `joblib.load()` in inference mode only. It was **not retrained or
  fine-tuned** at any point; `.fit()` is never called.
- OULAD predicted probabilities were generated fresh from the frozen
  model applied to `final_student_psychology_dataset.csv`, using the
  fixed seven-feature schema in the model's expected column order.
- ASSISTments predicted probabilities were loaded directly from the
  pre-computed `assistments_predictions.csv` file produced by
  `generate_external_predictions.py`, and were not regenerated here.
- All statistics below describe the **model's own output distribution**
  and nothing about the correctness of that output.

## Descriptive Statistics of Predicted Probabilities

{stats_table}

## Risk Group Distribution

Predicted probabilities were bucketed into three descriptive bands:
Low (0.00-0.30), Moderate (0.30-0.70), and High (0.70-1.00). These are
descriptive bins over the model's output, not evaluation thresholds.

### OULAD (training population)

{risk_table(oulad_risk_summary)}

### ASSISTments (external population)

{risk_table(assistments_risk_summary)}

## Fine-Grained Probability Decile Breakdown

For a more granular view of the output distribution than the three-band
risk grouping above, predicted probabilities were also binned into ten
equal-width deciles (0.00-0.10 through 0.90-1.00). This breakdown is
purely descriptive of the model's output and is provided to support
richer reporting tables.

{decile_table(oulad_decile_summary, assistments_decile_summary)}

## Descriptive Confidence Analysis

A descriptive confidence measure was computed as the distance of each
predicted probability from the 0.5 decision boundary:

```
confidence = abs(predicted_probability - 0.5)
```

Confidence ranges from 0.0 (maximally uncertain output, p = 0.5) to 0.5
(maximally decisive output, p = 0.0 or p = 1.0). **This measure
describes how decisive the model's output is -- it is not, and must
never be interpreted as, a measure of whether that output is correct.**
No ground-truth label is used anywhere in this computation.

### OULAD (training population)

Mean confidence: {oulad_confidence_summary['mean_confidence'].iloc[0]:.4f} |
Median confidence: {oulad_confidence_summary['median_confidence'].iloc[0]:.4f}

{confidence_table(oulad_confidence_summary)}

### ASSISTments (external population)

Mean confidence: {assistments_confidence_summary['mean_confidence'].iloc[0]:.4f} |
Median confidence: {assistments_confidence_summary['median_confidence'].iloc[0]:.4f}

{confidence_table(assistments_confidence_summary)}

## Figures

- `figures/predicted_probability_distribution.png` -- side-by-side histograms
  of predicted probabilities, OULAD vs. ASSISTments
- `figures/prediction_density_comparison.png` -- overlaid kernel-density
  comparison of predicted probabilities
- `figures/confidence_distribution.png` -- side-by-side histograms of the
  descriptive confidence measure, OULAD vs. ASSISTments

## Data Files

- `outputs/prediction_distribution_statistics.csv` -- mean, median, min,
  max, standard deviation, quartiles, and IQR for both populations
- `outputs/confidence_distribution.csv` -- low/medium/high confidence
  band counts and percentages for both populations
- `outputs/predicted_probability_deciles.csv` -- the ten-bin decile
  breakdown underlying the table above, for both populations

## Scope Note

This report and its underlying script intentionally exclude any
assessment of predictive correctness. Descriptive shifts in the
predicted-probability distribution, risk-group composition, or
confidence distribution between OULAD and ASSISTments are informative
about how the model's output behaves under population change, but are
an entirely separate question from whether those outputs are accurate
for the ASSISTments population, which this pipeline stage does not and
cannot assess without ground-truth labels.
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
    logger.info("EduGuard-AI :: Prediction Distribution Analysis (OULAD vs. ASSISTments)")
    logger.info("=" * 70)

    # ---- 1. Load the frozen model (inference mode only) ----
    model = load_model(MODEL_PATH)

    # ---- 2. Load OULAD dataset and generate fresh predictions ----
    oulad_df = load_oulad_dataset(OULAD_DATA_PATH)
    validate_feature_columns(oulad_df, "OULAD")
    oulad_probs = generate_oulad_predictions(model, oulad_df)

    # ---- 3. Load pre-computed ASSISTments predictions ----
    assistments_df = load_assistments_predictions(ASSISTMENTS_PREDICTIONS_PATH)
    assistments_probs = assistments_df["predicted_probability"]

    # ---- 4. Descriptive statistics ----
    oulad_stats = compute_descriptive_statistics(oulad_probs, "OULAD")
    assistments_stats = compute_descriptive_statistics(assistments_probs, "ASSISTments")
    statistics_df = pd.DataFrame([oulad_stats, assistments_stats])
    statistics_df.to_csv(STATISTICS_PATH, index=False)
    logger.info(f"[SAVED] {STATISTICS_PATH}")

    # ---- 5. Risk group distribution ----
    oulad_risk_summary = summarize_risk_groups(oulad_probs, "OULAD")
    assistments_risk_summary = summarize_risk_groups(assistments_probs, "ASSISTments")

    # ---- 6. Descriptive confidence measure ----
    oulad_confidence = compute_confidence(oulad_probs)
    assistments_confidence = compute_confidence(assistments_probs)
    oulad_confidence_summary = summarize_confidence(oulad_confidence, "OULAD")
    assistments_confidence_summary = summarize_confidence(assistments_confidence, "ASSISTments")

    confidence_export = pd.concat([oulad_confidence_summary, assistments_confidence_summary], ignore_index=True)
    confidence_export.to_csv(CONFIDENCE_PATH, index=False)
    logger.info(f"[SAVED] {CONFIDENCE_PATH}")

    # ---- 6b. Fine-grained probability decile breakdown ----
    oulad_decile_summary = compute_decile_distribution(oulad_probs, "OULAD")
    assistments_decile_summary = compute_decile_distribution(assistments_probs, "ASSISTments")
    decile_export = pd.concat([oulad_decile_summary, assistments_decile_summary], ignore_index=True)
    decile_export.to_csv(DECILE_PATH, index=False)
    logger.info(f"[SAVED] {DECILE_PATH}")

    # ---- 7. Figures ----
    logger.info("Generating publication-quality figures ...")
    plot_probability_distribution(oulad_probs, assistments_probs, PROBABILITY_DIST_FIG_PATH)
    plot_density_comparison(oulad_probs, assistments_probs, DENSITY_COMPARISON_FIG_PATH)
    plot_confidence_distribution(oulad_confidence, assistments_confidence, CONFIDENCE_DIST_FIG_PATH)

    # ---- 8. Markdown report ----
    generate_report(
        oulad_stats, assistments_stats,
        oulad_risk_summary, assistments_risk_summary,
        oulad_confidence_summary, assistments_confidence_summary,
        oulad_decile_summary, assistments_decile_summary,
        REPORT_PATH,
    )

    logger.info("=" * 70)
    logger.info("Prediction distribution analysis complete.")
    logger.info(
        f"OULAD mean predicted probability: {oulad_stats['mean']:.4f} | "
        f"ASSISTments mean predicted probability: {assistments_stats['mean']:.4f}"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
