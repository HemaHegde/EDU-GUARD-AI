"""
EduGuard-AI :: External Validation Pipeline
============================================
Script: feature_distribution_comparison.py

PURPOSE (strict scope)
-----------------------
Compare the behavioural feature distributions of the OULAD training
population against the reconstructed ASSISTments 2012-2013 external
validation population, for the seven fixed behavioural features used by
the EduGuard-AI risk model.

This is a COVARIATE-SHIFT ANALYSIS, not a model evaluation. It answers
"do the two populations occupy a similar feature space?", not "is the
model's prediction correct?". Accordingly, this script does NOT:
    - load the trained XGBoost model
    - generate predictions
    - compute SHAP values
    - compute or reference any accuracy-style metric
It works exclusively on the two raw behavioural feature tables.

Run:
    python feature_distribution_comparison.py

Paths are resolved automatically relative to this script's location, so
the script works regardless of where the repository is cloned. Only the
CONFIGURATION section below should need editing if your local file
layout differs.

Allowed dependencies: pandas, numpy, matplotlib, scipy, pathlib
(standard library `logging` is used for structured console output).
"""

from __future__ import annotations

import logging
from pathlib import Path

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
logger = logging.getLogger("feature_distribution_comparison")


# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/external_validation/feature_distribution_comparison.py
# so the project root is two directories above this file.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # ml/external_validation -> ml -> project_root

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# OULAD reference population (training-distribution behavioural profiles).
# NOTE: the observed on-disk filename for this artefact is
# 'final_student_psychology_dataset.csv' under ml/, not
# 'datasets/OULAD/final_student_profiles.csv'. If your local layout uses
# a different name/location, update OULAD_PATH accordingly -- no other
# part of the script needs to change.
OULAD_PATH = PROJECT_ROOT / "ml" / "final_student_psychology_dataset.csv"

# ASSISTments external validation population (reconstructed behavioural
# profiles produced by preprocess_assistments.py).
ASSISTMENTS_PATH = PROJECT_ROOT / "datasets" / "ASSISTments" / "assistments_student_profiles.csv"

# Output locations.
FIGURES_DIR = SCRIPT_DIR / "figures"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
REPORTS_DIR = SCRIPT_DIR / "reports"

STATISTICS_OUTPUT_PATH = OUTPUTS_DIR / "feature_distribution_statistics.csv"
SHIFT_SUMMARY_OUTPUT_PATH = OUTPUTS_DIR / "feature_shift_summary.csv"
REPORT_OUTPUT_PATH = REPORTS_DIR / "feature_distribution_report.md"

# The seven fixed behavioural features shared by both datasets.
FEATURES = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
]

# Number of quantile buckets used for the Population Stability Index.
# 10 buckets (deciles) is the conventional default in industry/academic
# PSI implementations, balancing granularity against per-bucket sample
# size.
PSI_BUCKETS = 10

# PSI interpretation thresholds (standard convention).
PSI_THRESHOLD_NEGLIGIBLE = 0.10
PSI_THRESHOLD_MODERATE = 0.25

# Figure resolution.
FIGURE_DPI = 300


# --------------------------------------------------------------------------
# STEP HELPERS
# --------------------------------------------------------------------------

def load_dataset(path: Path, label: str) -> pd.DataFrame:
    """Load a behavioural feature CSV and verify it is non-empty."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} dataset not found at: {path}\n"
            "Update the corresponding path constant in the CONFIGURATION "
            "section if your local file layout differs."
        )
    logger.info(f"Loading {label} dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {label} dataset: {len(df)} rows.")
    return df


def validate_feature_columns(df: pd.DataFrame, label: str) -> None:
    """Verify that all seven required behavioural feature columns exist."""
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            f"The {label} dataset is missing required feature column(s): {missing}. "
            f"Expected all of: {FEATURES}."
        )
    logger.info(f"All seven required feature columns verified in {label} dataset.")


def compute_descriptive_statistics(
    oulad_df: pd.DataFrame, assistments_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute descriptive statistics (sample size, mean, std, median, min,
    max, 25th/75th percentile) for each of the seven features, separately
    for OULAD and ASSISTments. Returns one row per (feature, dataset)
    combination.
    """
    records = []
    for feature in FEATURES:
        for dataset_name, df in (("OULAD", oulad_df), ("ASSISTments", assistments_df)):
            series = df[feature].dropna()
            records.append(
                {
                    "feature": feature,
                    "dataset": dataset_name,
                    "sample_size": int(series.shape[0]),
                    "mean": series.mean(),
                    "std": series.std(ddof=1),
                    "median": series.median(),
                    "min": series.min(),
                    "max": series.max(),
                    "p25": series.quantile(0.25),
                    "p75": series.quantile(0.75),
                }
            )
    stats_df = pd.DataFrame(records)
    logger.info("Descriptive statistics computed for all seven features.")
    return stats_df


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = PSI_BUCKETS) -> float:
    """
    Compute the Population Stability Index (PSI) between a reference
    ('expected', here OULAD) distribution and a comparison ('actual',
    here ASSISTments) distribution, using a standard quantile-binning
    implementation.

    Methodology:
    1. Quantile breakpoints are derived from the reference (expected)
       distribution, so that each reference bucket contains an equal
       proportion of the reference sample.
    2. The outer breakpoints are extended to -inf / +inf so that any
       comparison-distribution values falling outside the reference
       range are still captured rather than dropped.
    3. The proportion of each distribution falling into each bucket is
       computed, with a small epsilon added to avoid division-by-zero
       or log(0) for empty buckets.
    4. PSI = sum over buckets of (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    A higher PSI indicates greater distributional shift between the two
    populations for that feature.
    """
    epsilon = 1e-6

    # Quantile breakpoints based on the reference (expected) distribution.
    quantiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, quantiles)

    # Extend the outer edges to capture out-of-range comparison values.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Bucket both distributions using the same breakpoints.
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    # Avoid division-by-zero / log(0) for empty buckets.
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def interpret_psi(psi_value: float) -> str:
    """Map a PSI value to its standard qualitative interpretation."""
    if psi_value < PSI_THRESHOLD_NEGLIGIBLE:
        return "Negligible shift"
    elif psi_value < PSI_THRESHOLD_MODERATE:
        return "Moderate shift"
    else:
        return "Significant shift"


def compute_shift_summary(oulad_df: pd.DataFrame, assistments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, for each of the seven features, the two-sample
    Kolmogorov-Smirnov statistic and p-value, and the Population
    Stability Index (PSI), comparing OULAD (reference) to ASSISTments
    (comparison).
    """
    records = []
    for feature in FEATURES:
        oulad_values = oulad_df[feature].dropna().to_numpy()
        assistments_values = assistments_df[feature].dropna().to_numpy()

        ks_statistic, ks_pvalue = stats.ks_2samp(oulad_values, assistments_values)
        psi_value = calculate_psi(oulad_values, assistments_values)

        records.append(
            {
                "feature": feature,
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "psi": psi_value,
                "psi_interpretation": interpret_psi(psi_value),
            }
        )
        logger.info(
            f"Feature '{feature}': KS={ks_statistic:.4f} (p={ks_pvalue:.4g}), "
            f"PSI={psi_value:.4f} ({interpret_psi(psi_value)})"
        )

    shift_df = pd.DataFrame(records)
    return shift_df


def generate_overlaid_histograms(
    oulad_df: pd.DataFrame, assistments_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Generate one overlaid histogram per feature, showing the OULAD and
    ASSISTments distributions together, saved at 300 DPI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in FEATURES:
        oulad_values = oulad_df[feature].dropna()
        assistments_values = assistments_df[feature].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))

        # Shared bin edges across both distributions for a fair visual
        # comparison, derived from the combined value range.
        combined_min = min(oulad_values.min(), assistments_values.min())
        combined_max = max(oulad_values.max(), assistments_values.max())
        bin_edges = np.linspace(combined_min, combined_max, 40)

        ax.hist(
            oulad_values,
            bins=bin_edges,
            alpha=0.5,
            label=f"OULAD (n={len(oulad_values)})",
            density=True,
            color="#1f77b4",
        )
        ax.hist(
            assistments_values,
            bins=bin_edges,
            alpha=0.5,
            label=f"ASSISTments (n={len(assistments_values)})",
            density=True,
            color="#d62728",
        )

        ax.set_title(f"Distribution Comparison: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        output_path = output_dir / f"{feature}_distribution_comparison.png"
        fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)

        logger.info(f"Saved histogram for '{feature}' to: {output_path}")


def write_markdown_report(
    stats_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    oulad_df: pd.DataFrame,
    assistments_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a Markdown report summarising dataset sizes, descriptive
    statistics, and PSI-based covariate-shift interpretation for each
    feature. Explicitly scoped to distributional analysis only -- no
    conclusions about model accuracy are drawn or implied.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Feature Distribution Comparison Report")
    lines.append("")
    lines.append("`ml/external_validation/reports/feature_distribution_report.md`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report presents a covariate-shift analysis between the OULAD "
        "training population and the ASSISTments 2012-2013 external "
        "validation population, restricted to the seven fixed behavioural "
        "features used by the EduGuard-AI risk model. **This report makes "
        "no claims about model accuracy, prediction correctness, or "
        "classification performance.** It describes only whether the two "
        "populations occupy a similar behavioural feature space."
    )
    lines.append("")

    # --- Dataset sizes -----------------------------------------------
    lines.append("## Dataset Sizes")
    lines.append("")
    lines.append(f"- OULAD reference dataset: **{len(oulad_df)}** rows")
    lines.append(f"- ASSISTments external dataset: **{len(assistments_df)}** rows")
    lines.append("")

    # --- Descriptive statistics ---------------------------------------
    lines.append("## Descriptive Statistics")
    lines.append("")
    lines.append(
        "Sample size, mean, standard deviation, median, minimum, maximum, "
        "and interquartile bounds for each feature, reported separately "
        "for each dataset."
    )
    lines.append("")
    lines.append(stats_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # --- PSI / KS summary ------------------------------------------------
    lines.append("## Distributional Shift Summary (KS Test and PSI)")
    lines.append("")
    lines.append(
        "The two-sample Kolmogorov-Smirnov (KS) test evaluates whether the "
        "OULAD and ASSISTments distributions for a feature differ "
        "significantly. The Population Stability Index (PSI) quantifies "
        "the magnitude of that shift using the following standard "
        "interpretation thresholds:"
    )
    lines.append("")
    lines.append(f"- PSI < {PSI_THRESHOLD_NEGLIGIBLE:.2f} -- Negligible shift")
    lines.append(
        f"- {PSI_THRESHOLD_NEGLIGIBLE:.2f} <= PSI < {PSI_THRESHOLD_MODERATE:.2f} -- Moderate shift"
    )
    lines.append(f"- PSI >= {PSI_THRESHOLD_MODERATE:.2f} -- Significant shift")
    lines.append("")
    lines.append(shift_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # --- Interpretation notes ------------------------------------------
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append(
        "- A **negligible or moderate** PSI for a feature suggests that "
        "ASSISTments students occupy a broadly comparable region of that "
        "feature's distribution to the OULAD training population, which "
        "is a precondition for the trained model's behavioural decision "
        "logic to be meaningfully applicable to the external population."
    )
    lines.append(
        "- A **significant** PSI for a feature indicates that the two "
        "populations differ substantially along that dimension, which "
        "should be considered when interpreting any downstream prediction "
        "or SHAP-based analysis for that feature."
    )
    lines.append(
        "- These results describe **covariate shift only**. They do not, "
        "and cannot, indicate whether the model's predictions on "
        "ASSISTments are correct, since no ground-truth psychological "
        "risk label exists for that population."
    )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown report written to: {output_path}")


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    # ---- Load both datasets ----
    oulad_df = load_dataset(OULAD_PATH, "OULAD")
    assistments_df = load_dataset(ASSISTMENTS_PATH, "ASSISTments")

    # ---- Validate required feature columns in both datasets ----
    validate_feature_columns(oulad_df, "OULAD")
    validate_feature_columns(assistments_df, "ASSISTments")

    # ---- Compute descriptive statistics ----
    stats_df = compute_descriptive_statistics(oulad_df, assistments_df)

    # ---- Compute KS test and PSI shift summary ----
    shift_df = compute_shift_summary(oulad_df, assistments_df)

    # ---- Generate overlaid histograms (one per feature) ----
    generate_overlaid_histograms(oulad_df, assistments_df, FIGURES_DIR)

    # ---- Save statistics and shift summary CSVs ----
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(STATISTICS_OUTPUT_PATH, index=False)
    logger.info(f"Descriptive statistics saved to: {STATISTICS_OUTPUT_PATH}")

    shift_df.to_csv(SHIFT_SUMMARY_OUTPUT_PATH, index=False)
    logger.info(f"Feature shift summary saved to: {SHIFT_SUMMARY_OUTPUT_PATH}")

    # ---- Write the Markdown report ----
    write_markdown_report(stats_df, shift_df, oulad_df, assistments_df, REPORT_OUTPUT_PATH)

    # ---- Final console summary ----
    logger.info("=" * 60)
    logger.info("EduGuard-AI :: Feature distribution comparison complete")
    logger.info("=" * 60)
    logger.info(f"OULAD rows:        {len(oulad_df)}")
    logger.info(f"ASSISTments rows:  {len(assistments_df)}")
    logger.info(f"Features compared: {len(FEATURES)}")
    logger.info(
        f"Significant-shift features: "
        f"{shift_df.loc[shift_df['psi_interpretation'] == 'Significant shift', 'feature'].tolist()}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
