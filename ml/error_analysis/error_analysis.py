"""
EduGuard-AI :: Error Analysis Pipeline
========================================
Script: error_analysis.py

PURPOSE (strict scope)
-----------------------
Perform a complete error analysis of the frozen, already-trained OULAD
XGBoost psychological risk model (`academic_risk_xgboost.pkl`) on its
own held-out OULAD test partition.

This is an ANALYSIS script only. The model is loaded strictly in
inference mode via `predict()` / `predict_proba()`. It is NEVER
retrained, refit, or fine-tuned in any way -- `.fit()` is not called
anywhere in this script.

Scope is deliberately limited to:
    - re-deriving the SAME held-out test split used at training time
      (random_state=42, test_size=0.20, stratify=y), so that error
      analysis is performed on genuinely unseen rows
    - generating predicted labels and predicted probabilities for that
      test partition
    - standard label-dependent evaluation metrics (accuracy, precision,
      recall, F1, ROC-AUC, confusion matrix) computed against the real
      OULAD ground truth (`psychological_risk`), which -- unlike the
      ASSISTments external-validation stage -- is legitimately available
      here
    - itemising false positives, false negatives, and the lowest-
      confidence ("hardest") predictions for qualitative inspection

Explicitly OUT of scope for this script:
    - retraining or refitting the model in any way
    - SHAP / feature-importance explainability (see preprocess_oulad.py)
    - external validation on ASSISTments or any other population
      (see generate_external_predictions.py)
    - computational / resource benchmarking (see computational_benchmarking.py)

Run:
    python error_analysis.py

Only the model/input paths below may need editing; everything else is
self-contained. Paths are resolved automatically relative to this
script's location, so the script works regardless of where the
repository is cloned.

Allowed dependencies: pandas, numpy, joblib, matplotlib, scikit-learn
(train_test_split + metrics only), pathlib (standard library `logging`
for structured console output). No seaborn. No retraining. No SHAP.
No external validation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: safe for headless / CI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
# A simple, timestamped console logger. Using `logging` (rather than bare
# `print`) gives clearer severity levels (INFO / ERROR) and consistent
# formatting, matching the other pipeline scripts.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("error_analysis")


# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/error_analysis/error_analysis.py
# so the project root is two directories above this file. Resolving it
# from __file__ (rather than the current working directory) ensures the
# script behaves identically no matter where it is invoked from, and
# avoids any hardcoded, OS-specific paths.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # ml/error_analysis -> ml -> project_root

# Input: OULAD behavioural feature table + psychological_risk ground
# truth, produced by preprocess_oulad.py.
DATA_PATH = PROJECT_ROOT / "ml" / "final_student_psychology_dataset.csv"

# Model: the frozen, already-trained OULAD XGBoost psychological risk
# classifier. Loaded via joblib and used strictly for inference.
MODEL_PATH = PROJECT_ROOT / "ml" / "academic_risk_xgboost.pkl"

# Output locations.
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
REPORT_DIR = SCRIPT_DIR / "reports"

CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.csv"
FALSE_POSITIVE_PATH = OUTPUT_DIR / "false_positive_cases.csv"
FALSE_NEGATIVE_PATH = OUTPUT_DIR / "false_negative_cases.csv"
HARDEST_CASES_PATH = OUTPUT_DIR / "hardest_cases.csv"

CONFUSION_MATRIX_FIG_PATH = FIGURE_DIR / "confusion_matrix.png"
PROBABILITY_HIST_FIG_PATH = FIGURE_DIR / "prediction_probability_histogram.png"
ERROR_DISTRIBUTION_FIG_PATH = FIGURE_DIR / "error_probability_distribution.png"

REPORT_PATH = REPORT_DIR / "error_analysis_report.md"

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

TARGET_COLUMN = "psychological_risk"

# Same split parameters used in preprocess_oulad.py's train/test split.
# Re-applying them here (without ever calling .fit()) recovers the exact
# same held-out test partition the model was originally evaluated on.
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Decision threshold used to derive a binary predicted_label from the
# continuous predicted_probability. 0.5 matches the threshold implicit
# in the OULAD training pipeline's use of `.predict()`.
DECISION_THRESHOLD = 0.5

# Number of lowest-confidence predictions to retain for qualitative
# "hardest cases" inspection.
N_HARDEST_CASES = 50


# --------------------------------------------------------------------------
# STEP HELPERS
# --------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load the OULAD behavioural feature table with ground-truth labels."""
    if not path.exists():
        raise FileNotFoundError(
            f"OULAD dataset not found at: {path}\n"
            "Ensure preprocess_oulad.py has been run and produced "
            "final_student_psychology_dataset.csv at the expected location."
        )
    logger.info(f"Loading OULAD dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows.")
    return df


def load_model(path: Path):
    """Load the frozen, pre-trained OULAD XGBoost model via joblib."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at: {path}\n"
            "Ensure academic_risk_xgboost.pkl has been generated by "
            "preprocess_oulad.py and placed at the expected location."
        )
    logger.info(f"Loading trained OULAD XGBoost model from: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    return model


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verify that all seven required behavioural feature columns and the
    target column are present. Raises a clear, actionable error listing
    exactly which columns are missing if validation fails.
    """
    missing_features = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            "The input dataset is missing required model feature column(s): "
            f"{missing_features}. Expected all of: {REQUIRED_FEATURE_COLUMNS}."
        )
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"The input dataset is missing the required target column "
            f"'{TARGET_COLUMN}'."
        )
    logger.info("All required feature columns and the target column are present.")


def recover_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recover the SAME held-out test partition used during original
    training, by re-applying an identical stratified split
    (random_state=42, test_size=0.20) to the full feature/target
    definitions. No fitting occurs here -- this only reproduces which
    rows were held out.

    Returns the test-partition rows (original dataframe subset,
    including all original columns) so hardest-case / FP / FN exports
    can retain full row context.
    """
    X = df[REQUIRED_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    logger.info(
        f"Recovering held-out test split (test_size={TEST_SIZE}, "
        f"random_state={RANDOM_STATE}, stratify=y)..."
    )
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_df = df.loc[X_test.index].copy()
    logger.info(f"Held-out test partition recovered: {len(test_df)} rows.")
    return test_df


def generate_predictions(model, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the frozen model to the recovered test partition and append
    predicted_probability and predicted_label. No fitting occurs here.
    """
    X_test = test_df[REQUIRED_FEATURE_COLUMNS]

    logger.info("Generating predicted probabilities using the frozen XGBoost model...")
    probabilities = model.predict_proba(X_test)[:, 1]
    labels = (probabilities >= DECISION_THRESHOLD).astype(int)

    result = test_df.copy()
    result["predicted_probability"] = probabilities
    result["predicted_label"] = labels
    result["confidence_margin"] = (result["predicted_probability"] - 0.5).abs()

    logger.info(f"Predictions generated for {len(result)} test rows.")
    return result


def compute_metrics(result: pd.DataFrame) -> dict:
    """Compute standard label-dependent evaluation metrics against the
    real OULAD ground truth, plus derive the confusion-matrix layout."""
    y_true = result[TARGET_COLUMN]
    y_pred = result["predicted_label"]
    y_prob = result["predicted_probability"]

    logger.info("Computing evaluation metrics against OULAD ground truth...")

    metrics = {
        "n_test_rows": int(len(result)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics.update(
        {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    )

    logger.info(
        "Metrics -- accuracy: {accuracy:.4f} | precision: {precision:.4f} | "
        "recall: {recall:.4f} | f1: {f1:.4f} | roc_auc: {roc_auc:.4f}".format(**metrics)
    )
    logger.info(
        f"Confusion matrix -- TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}"
    )

    return metrics


def extract_error_cases(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract:
        - every false-positive row (predicted 1, true 0)
        - every false-negative row (predicted 0, true 1)
        - the N lowest-confidence predictions (smallest |probability - 0.5|)

    Each exported row retains: predicted_probability, ground truth
    (psychological_risk), and all seven model features (plus any other
    original columns, for full context).
    """
    export_columns = (
        REQUIRED_FEATURE_COLUMNS
        + [TARGET_COLUMN, "predicted_label", "predicted_probability"]
    )
    # Preserve any additional identifying/original columns not already
    # in export_columns (e.g. id_student), placed first for readability.
    other_columns = [c for c in result.columns if c not in export_columns + ["confidence_margin"]]
    ordered_columns = other_columns + export_columns

    false_positives = result[
        (result["predicted_label"] == 1) & (result[TARGET_COLUMN] == 0)
    ][ordered_columns].copy()

    false_negatives = result[
        (result["predicted_label"] == 0) & (result[TARGET_COLUMN] == 1)
    ][ordered_columns].copy()

    hardest_cases = (
        result.sort_values("confidence_margin", ascending=True)
        .head(N_HARDEST_CASES)[ordered_columns]
        .copy()
    )

    logger.info(f"False positives identified: {len(false_positives)}")
    logger.info(f"False negatives identified: {len(false_negatives)}")
    logger.info(f"Hardest cases retained: {len(hardest_cases)}")

    return false_positives, false_negatives, hardest_cases


# --------------------------------------------------------------------------
# FIGURE HELPERS
# --------------------------------------------------------------------------

def plot_confusion_matrix(result: pd.DataFrame, path: Path) -> None:
    """Render and save a publication-quality confusion matrix figure."""
    y_true = result[TARGET_COLUMN]
    y_pred = result["predicted_label"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Low Risk (0)", "High Risk (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix -- OULAD Held-Out Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to: {path}")


def plot_probability_histogram(result: pd.DataFrame, path: Path) -> None:
    """Histogram of predicted probabilities, split by ground-truth class."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    bins = np.linspace(0, 1, 41)
    for cls, label, color in [(0, "True Low Risk (0)", "#4C72B0"), (1, "True High Risk (1)", "#C44E52")]:
        subset = result.loc[result[TARGET_COLUMN] == cls, "predicted_probability"]
        ax.hist(subset, bins=bins, alpha=0.6, label=label, color=color, edgecolor="black", linewidth=0.3)

    ax.axvline(DECISION_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"Decision threshold ({DECISION_THRESHOLD})")
    ax.set_xlabel("Predicted Probability (psychological_risk = 1)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution by Ground-Truth Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved prediction probability histogram to: {path}")


def plot_error_probability_distribution(result: pd.DataFrame, path: Path) -> None:
    """Histogram comparing predicted probabilities for correct vs. incorrect predictions."""
    correct = result[result[TARGET_COLUMN] == result["predicted_label"]]["predicted_probability"]
    incorrect = result[result[TARGET_COLUMN] != result["predicted_label"]]["predicted_probability"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bins = np.linspace(0, 1, 41)
    ax.hist(correct, bins=bins, alpha=0.6, label=f"Correct (n={len(correct)})", color="#55A868", edgecolor="black", linewidth=0.3)
    ax.hist(incorrect, bins=bins, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", color="#C44E52", edgecolor="black", linewidth=0.3)

    ax.axvline(DECISION_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"Decision threshold ({DECISION_THRESHOLD})")
    ax.set_xlabel("Predicted Probability (psychological_risk = 1)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution: Correct vs. Incorrect Predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved error probability distribution figure to: {path}")


# --------------------------------------------------------------------------
# OUTPUT WRITERS
# --------------------------------------------------------------------------

def save_classification_report(result: pd.DataFrame, path: Path) -> None:
    """Save sklearn's classification_report as a tidy CSV."""
    y_true = result[TARGET_COLUMN]
    y_pred = result["predicted_label"]

    report_dict = classification_report(
        y_true, y_pred, target_names=["Low Risk (0)", "High Risk (1)"], output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(path, index=True)
    logger.info(f"Saved classification report to: {path}")


def save_confusion_matrix(metrics: dict, path: Path) -> None:
    """Save the confusion matrix counts as a labeled CSV."""
    cm_df = pd.DataFrame(
        [
            {"actual": "Low Risk (0)", "predicted_low_risk_0": metrics["true_negatives"], "predicted_high_risk_1": metrics["false_positives"]},
            {"actual": "High Risk (1)", "predicted_low_risk_0": metrics["false_negatives"], "predicted_high_risk_1": metrics["true_positives"]},
        ]
    )
    cm_df.to_csv(path, index=False)
    logger.info(f"Saved confusion matrix to: {path}")


def write_report(
    metrics: dict,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    hardest_cases: pd.DataFrame,
) -> None:
    """Write the human-readable Markdown error analysis report."""
    n_fp = len(false_positives)
    n_fn = len(false_negatives)

    if n_fp > n_fn:
        larger_source = (
            f"False positives ({n_fp}) outnumber false negatives ({n_fn}). "
            "The model more often over-predicts psychological risk (flagging "
            "students as high risk who ultimately passed) than it misses "
            "genuinely at-risk students."
        )
    elif n_fn > n_fp:
        larger_source = (
            f"False negatives ({n_fn}) outnumber false positives ({n_fp}). "
            "The model more often under-predicts psychological risk (missing "
            "students who failed or withdrew) than it raises unwarranted "
            "alerts."
        )
    else:
        larger_source = (
            f"False positives and false negatives are balanced ({n_fp} each)."
        )

    hardest_summary = (
        f"The {len(hardest_cases)} hardest cases -- those with predicted "
        "probability closest to the 0.5 decision threshold -- represent "
        "students whose behavioural feature profile placed them near the "
        "model's decision boundary. These are the rows most likely to flip "
        "label under small changes in input features or threshold, and are "
        "the natural starting point for qualitative review or threshold "
        "tuning."
    )

    lines = [
        "# Error Analysis Report -- OULAD Psychological Risk Model",
        "",
        "## Scope",
        "",
        "This report analyzes the frozen, already-trained OULAD XGBoost "
        "psychological risk model on its own held-out OULAD test "
        "partition. The model was not retrained, refit, or modified in "
        "any way as part of this analysis.",
        "",
        "## Dataset and Split",
        "",
        f"- Full dataset size (all OULAD rows): loaded from "
        f"`final_student_psychology_dataset.csv`",
        f"- Held-out test partition size: **{metrics['n_test_rows']}** rows",
        f"- Split parameters: `test_size=0.20`, `random_state=42`, `stratify=y` "
        "(identical to the split used in `preprocess_oulad.py`)",
        "",
        "## Evaluation Metrics (Held-Out Test Set)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 Score | {metrics['f1']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "| | Predicted Low Risk (0) | Predicted High Risk (1) |",
        "|---|---|---|",
        f"| **Actual Low Risk (0)** | {metrics['true_negatives']} (TN) | {metrics['false_positives']} (FP) |",
        f"| **Actual High Risk (1)** | {metrics['false_negatives']} (FN) | {metrics['true_positives']} (TP) |",
        "",
        f"- Number of false positives: **{n_fp}**",
        f"- Number of false negatives: **{n_fn}**",
        "",
        "## Discussion: Largest Source of Errors",
        "",
        larger_source,
        "",
        "## Hardest Cases Summary",
        "",
        hardest_summary,
        "",
        "## Output Files",
        "",
        "- `outputs/classification_report.csv` -- full per-class precision/recall/F1",
        "- `outputs/confusion_matrix.csv` -- confusion matrix counts",
        "- `outputs/false_positive_cases.csv` -- every false-positive row (all features + probability)",
        "- `outputs/false_negative_cases.csv` -- every false-negative row (all features + probability)",
        "- `outputs/hardest_cases.csv` -- the 50 lowest-confidence predictions",
        "- `figures/confusion_matrix.png`",
        "- `figures/prediction_probability_histogram.png`",
        "- `figures/error_probability_distribution.png`",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved error analysis report to: {REPORT_PATH}")


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    try:
        # ---- 1. Load OULAD dataset and frozen model ----
        df = load_dataset(DATA_PATH)
        model = load_model(MODEL_PATH)

        # ---- 2. Validate schema ----
        validate_columns(df)

        # ---- 3. Recover the original held-out test split (no fitting) ----
        test_df = recover_test_split(df)

        # ---- 4. Generate predictions on the test partition ----
        result = generate_predictions(model, test_df)

        # ---- 5. Compute evaluation metrics ----
        metrics = compute_metrics(result)

        # ---- 6. Extract false positives / false negatives / hardest cases ----
        false_positives, false_negatives, hardest_cases = extract_error_cases(result)

        # ---- 7. Write CSV outputs ----
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_classification_report(result, CLASSIFICATION_REPORT_PATH)
        save_confusion_matrix(metrics, CONFUSION_MATRIX_PATH)
        false_positives.to_csv(FALSE_POSITIVE_PATH, index=False)
        logger.info(f"Saved false positive cases to: {FALSE_POSITIVE_PATH}")
        false_negatives.to_csv(FALSE_NEGATIVE_PATH, index=False)
        logger.info(f"Saved false negative cases to: {FALSE_NEGATIVE_PATH}")
        hardest_cases.to_csv(HARDEST_CASES_PATH, index=False)
        logger.info(f"Saved hardest cases to: {HARDEST_CASES_PATH}")

        # ---- 8. Generate figures ----
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(result, CONFUSION_MATRIX_FIG_PATH)
        plot_probability_histogram(result, PROBABILITY_HIST_FIG_PATH)
        plot_error_probability_distribution(result, ERROR_DISTRIBUTION_FIG_PATH)

        # ---- 9. Write Markdown report ----
        write_report(metrics, false_positives, false_negatives, hardest_cases)

        logger.info("Error analysis completed successfully.")

    except (FileNotFoundError, ValueError) as exc:
        # Fail loudly and clearly rather than proceeding with invalid
        # or incomplete data / model state.
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
