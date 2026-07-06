"""
EduGuard-AI :: External Validation Pipeline
============================================
Script: generate_external_predictions.py

PURPOSE (strict scope)
-----------------------
Apply the already-trained OULAD XGBoost psychological risk model to the
reconstructed ASSISTments 2012-2013 behavioural feature table
(`assistments_student_profiles.csv`) and produce per-row predictions.

This script is a PREDICTION step only. It is NOT an evaluation script.

ASSISTments does not contain an equivalent of OULAD's `final_result`
(Pass / Fail / Withdrawn), and therefore no `psychological_risk` ground
truth exists for this population. Consequently this script deliberately
does NOT compute, import, or reference:
    - accuracy
    - precision / recall / F1
    - ROC-AUC / confusion matrices
    - calibration metrics
    - any other label-dependent evaluation metric
Any such computation would be scientifically invalid without ground
truth and is intentionally out of scope for this stage of the pipeline.
Model outputs are described only in terms of their own distribution.

REVIEWER-REQUESTED EXTENSION (supervised ground-truth evaluation)
-------------------------------------------------------------------
Reviewers requested a TRUE supervised evaluation against real
ASSISTments ground truth, rather than only the distribution-only
summary above. The raw ASSISTments export DOES contain a real,
observed ground-truth column, `correct` (whether a student answered a
given problem correctly) — this was NOT available as a per-row signal
in the prediction step before, and nothing about it is invented here.

IMPORTANT — GRANULARITY AND WHAT IS ACTUALLY BEING EVALUATED:
The trained model predicts per (student, class) psychological risk,
not per-interaction correctness, so `correct` cannot be used as a
row-level target directly. `preprocess_assistments.py` (UNCHANGED,
not modified by this extension) already aggregates `correct` into
`avg_score = mean(correct) * 100` at exactly the same
(student_class_id, student_id) granularity as the model's own
features and predictions. This script reuses that already-computed,
already-disclosed aggregate (per requirement #2: "use the
already-generated engineered features wherever possible") rather than
re-deriving anything from the raw file, and defines the supervised
ground-truth label as a disclosed, deterministic threshold on it:

    ground_truth_risk = 1  if avg_score < 50   else 0

50 is not a new/arbitrary cutoff invented for this evaluation — it is
the exact threshold `risk_service.py` already uses elsewhere in this
project to flag "Academic performance weakening" (see its
`if avg_score < 50` reason rule), so this reuses an existing,
already-justified decision boundary rather than introducing a new one.

This is a real, observed label derived deterministically from the
real `correct` column — NOT a synthetic label, NOT weak supervision,
and NOT fabricated. It is, however, NOT equivalent to OULAD's original
`psychological_risk` label (which came from `final_result`, a column
completely independent of the 7 model input features). Here, the
ground-truth label is built from `avg_score`, which is ALSO one of the
7 input features the model was trained on. This creates a genuine
methodological limitation (feature/label overlap) that is NOT hidden —
it is documented prominently in the generated markdown report's
Limitations section. Reviewers/readers should weigh this evaluation
accordingly: it is a legitimate, non-fabricated supervised check, but
not as independent as the original OULAD held-out evaluation.

This extension adds four NEW, additive outputs and touches no
existing function's behaviour:
    outputs/assistments_ground_truth_metrics.csv
    figures/assistments_confusion_matrix.png
    figures/assistments_roc_curve.png
    reports/assistments_ground_truth_validation.md
All original functions (load_student_profiles, load_model,
validate_feature_columns, generate_predictions,
log_prediction_summary) are UNCHANGED. `main()` gains additional
steps, appended after the existing ones, not replacing them.

Run:
    python generate_external_predictions.py

Only the model/input paths below may need editing; everything else is
self-contained. Paths are resolved automatically relative to this
script's location, so the script works regardless of where the
repository is cloned.

Allowed dependencies: pandas, numpy, joblib, pathlib (standard library
`logging` is used for structured console output). This extension adds
scikit-learn's metrics module (already a project dependency — see
preprocess_oulad.py) and matplotlib (Agg backend, headless-safe) for
the two required figures.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# SPRINT (reviewer extension) — supervised ground-truth evaluation.
# scikit-learn is already a project dependency (see preprocess_oulad.py,
# which already imports sklearn.metrics for the original OULAD
# evaluation). matplotlib is used only for the two required static
# figures; the Agg backend is selected explicitly BEFORE importing
# pyplot so this script runs headless (no display) in CI/servers.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
# A simple, timestamped console logger. Using `logging` (rather than bare
# `print`) gives clearer severity levels (INFO / ERROR) and consistent
# formatting, which is preferable for a script intended to run as part of
# a reproducible research pipeline.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_external_predictions")


# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/external_validation/generate_external_predictions.py
# so the project root is two directories above this file. Resolving it
# from __file__ (rather than the current working directory) ensures the
# script behaves identically no matter where it is invoked from, and
# avoids any hardcoded, OS-specific paths.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # ml/external_validation -> ml -> project_root

# Input: reconstructed ASSISTments behavioural feature table produced by
# preprocess_assistments.py.
INPUT_PATH = PROJECT_ROOT / "datasets" / "ASSISTments" / "assistments_student_profiles.csv"

# Model: the OULAD-trained XGBoost psychological risk classifier, frozen
# and reused as-is for external prediction (no retraining, no fine-tuning).
MODEL_PATH = PROJECT_ROOT / "ml" / "academic_risk_xgboost.pkl"

# Output: per-student predictions, written alongside the original
# behavioural profile columns.
OUTPUT_PATH = PROJECT_ROOT / "ml" / "external_validation" / "outputs" / "assistments_predictions.csv"

# --------------------------------------------------------------------------
# SPRINT (reviewer extension) — supervised ground-truth evaluation paths
# --------------------------------------------------------------------------
# New, additive output locations (per spec), all under the same
# ml/external_validation/ root as the existing OUTPUT_PATH above.
GROUND_TRUTH_METRICS_PATH = (
    PROJECT_ROOT / "ml" / "external_validation" / "outputs" / "assistments_ground_truth_metrics.csv"
)
CONFUSION_MATRIX_FIG_PATH = (
    PROJECT_ROOT / "ml" / "external_validation" / "figures" / "assistments_confusion_matrix.png"
)
ROC_CURVE_FIG_PATH = (
    PROJECT_ROOT / "ml" / "external_validation" / "figures" / "assistments_roc_curve.png"
)
MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT / "ml" / "external_validation" / "reports" / "assistments_ground_truth_validation.md"
)

# Ground-truth label threshold on avg_score (itself = mean(correct)*100,
# already computed by the UNCHANGED preprocess_assistments.py). This is
# NOT a new/arbitrary cutoff — it is the exact threshold risk_service.py
# already uses to flag "Academic performance weakening" elsewhere in
# this project (see that file's `if avg_score < 50` rule), reused here
# rather than inventing a new decision boundary. See the module
# docstring for the full disclosure of this label's construction and
# its documented limitation (feature/label overlap with avg_score).
#
# THIS IS THE SINGLE, SOLE CONFIGURATION POINT for the ground-truth
# threshold used throughout this script (label derivation, the metrics
# CSV, the console summary, and the markdown report all read this one
# constant — nothing else needs to change to explore a different
# cutoff). The default of 50.0 is unchanged and is what every generated
# output below reflects; see the markdown report's "Threshold
# Sensitivity Note" section for why this is not swapped out here.
GROUND_TRUTH_AVG_SCORE_THRESHOLD = 50.0

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

# Decision threshold used to derive a binary predicted_label from the
# continuous predicted_probability. 0.5 is the conventional default for
# a binary classifier and matches the threshold implicit in the OULAD
# training pipeline's use of `.predict()` / class-1 probability.
DECISION_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# STEP HELPERS
# --------------------------------------------------------------------------

def load_student_profiles(path: Path) -> pd.DataFrame:
    """Load the reconstructed ASSISTments student-class behavioural profiles."""
    if not path.exists():
        raise FileNotFoundError(
            f"ASSISTments student profile file not found at: {path}\n"
            "Run preprocess_assistments.py first to generate this file."
        )
    logger.info(f"Loading ASSISTments student profiles from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} student-class observation rows.")
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


def validate_feature_columns(df: pd.DataFrame) -> None:
    """
    Verify that all seven required behavioural feature columns are present
    in the input dataframe. Raises a clear, actionable error listing
    exactly which columns are missing if validation fails.
    """
    missing_columns = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The input dataset is missing required model feature column(s): "
            f"{missing_columns}. "
            f"Expected all of: {REQUIRED_FEATURE_COLUMNS}. "
            "Verify that preprocess_assistments.py completed successfully "
            "and produced the full seven-feature schema."
        )
    logger.info("All seven required feature columns are present and verified.")


def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the frozen model to the verified feature columns and append
    predicted_probability and predicted_label to the dataframe.

    Only prediction is performed here -- no metric requiring ground
    truth is computed, by design.
    """
    # Enforce exact column order expected by the model (matches the
    # training-time column order in preprocess_oulad.py's X definition).
    X = df[REQUIRED_FEATURE_COLUMNS]

    logger.info("Generating predicted probabilities using the frozen XGBoost model...")
    # predict_proba returns an (n_samples, n_classes) array; column index 1
    # corresponds to the positive class (psychological_risk = 1), matching
    # the label convention established during OULAD training.
    probabilities = model.predict_proba(X)[:, 1]

    # Binary label derived from the fixed decision threshold. This is a
    # deterministic transformation of the probability, not an evaluation.
    labels = (probabilities >= DECISION_THRESHOLD).astype(int)

    result = df.copy()
    result["predicted_probability"] = probabilities
    result["predicted_label"] = labels

    logger.info(f"Predictions generated for {len(result)} rows.")
    return result


def log_prediction_summary(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics of the generated predictions.

    These are purely descriptive summaries of the model's own output
    distribution -- they do not assess correctness, and are not to be
    interpreted as performance metrics.
    """
    probs = df["predicted_probability"]
    labels = df["predicted_label"]

    n_records = len(df)
    n_low_risk = int((labels == 0).sum())
    n_high_risk = int((labels == 1).sum())

    logger.info("=" * 60)
    logger.info("EduGuard-AI :: ASSISTments external prediction summary")
    logger.info("=" * 60)
    logger.info(f"Number of records scored:         {n_records}")
    logger.info(f"Decision threshold:                {DECISION_THRESHOLD:.2f}")
    logger.info("Predicted probability statistics:")
    logger.info(f"  - minimum:                       {probs.min():.6f}")
    logger.info(f"  - maximum:                        {probs.max():.6f}")
    logger.info(f"  - mean:                            {probs.mean():.6f}")
    logger.info(f"  - standard deviation:              {probs.std(ddof=0):.6f}")
    logger.info(f"Number predicted low risk (label=0): {n_low_risk}")
    logger.info(f"Number predicted high risk (label=1):{n_high_risk}")
    logger.info("=" * 60)


# --------------------------------------------------------------------------
# SPRINT (REVIEWER EXTENSION): SUPERVISED GROUND-TRUTH EVALUATION
# --------------------------------------------------------------------------
# Everything below is ADDITIVE. It does not alter any function above,
# does not change what generate_predictions() computes, and does not
# touch preprocess_assistments.py. See the module docstring for the
# full disclosure of how the ground-truth label is constructed and its
# documented limitation.

def derive_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives `ground_truth_risk` from the already-computed `avg_score`
    column (itself = mean(real ASSISTments `correct` values) * 100,
    computed by the unmodified preprocess_assistments.py).

    ground_truth_risk = 1 if avg_score < GROUND_TRUTH_AVG_SCORE_THRESHOLD else 0

    This is a deterministic transform of REAL, observed data — no
    value is invented, imputed as a label, or probabilistically
    guessed. See the module docstring for why this is not fabrication
    / weak supervision, and for the documented feature/label-overlap
    limitation this introduces.
    """
    if "avg_score" not in df.columns:
        raise ValueError(
            "Cannot derive ground-truth labels: 'avg_score' column is "
            "missing from the student profile table. This column is "
            "required (it is one of the seven fixed model features) "
            "and should already be present from preprocess_assistments.py."
        )

    result = df.copy()
    result["ground_truth_risk"] = (
        result["avg_score"] < GROUND_TRUTH_AVG_SCORE_THRESHOLD
    ).astype(int)
    return result


def compute_supervised_metrics(df: pd.DataFrame) -> dict:
    """
    Computes Accuracy, Precision, Recall, F1, ROC-AUC, and the
    confusion matrix comparing the model's existing
    predicted_label / predicted_probability against the real
    ground_truth_risk column derived above.

    Uses zero_division=0 for precision/recall/F1 so a degenerate case
    (e.g. no predicted positives) reports 0.0 rather than raising.
    ROC-AUC requires both classes to be present in the ground truth;
    if only one class exists, this is logged as a warning and
    roc_auc is reported as None (not fabricated as 0.0 or 1.0).
    """
    y_true = df["ground_truth_risk"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    y_prob = df["predicted_probability"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    n_records = len(df)
    # Class-distribution percentages, added for reporting in the console
    # summary and markdown report (per reviewer request). Purely
    # descriptive of the already-derived ground_truth_risk column —
    # nothing new is computed or labeled here.
    positive_pct = (n_positive / n_records * 100.0) if n_records > 0 else 0.0
    negative_pct = (n_negative / n_records * 100.0) if n_records > 0 else 0.0

    roc_auc = None
    fpr, tpr = None, None
    if n_positive > 0 and n_negative > 0:
        roc_auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    else:
        logger.warning(
            "ROC-AUC could not be computed: ground truth contains only "
            "one class (n_positive=%d, n_negative=%d) given threshold "
            "avg_score < %.1f. Reporting roc_auc=None rather than a "
            "fabricated value.",
            n_positive, n_negative, GROUND_TRUTH_AVG_SCORE_THRESHOLD,
        )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "n_records": len(df),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Saves a labeled, IEEE-figure-ready confusion matrix as a PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
    im = ax.imshow(cm, cmap="Blues")

    class_labels = ["Low Risk (0)", "High Risk (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground-truth label (avg_score-derived)")
    ax.set_title("ASSISTments External Validation\nConfusion Matrix")

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Confusion matrix figure saved to: {output_path}")


def plot_roc_curve(fpr, tpr, roc_auc: float, output_path: Path) -> None:
    """Saves an IEEE-figure-ready ROC curve as a PNG. No-op (with a
    warning) if roc_auc/fpr/tpr are None (single-class ground truth)."""
    if fpr is None or tpr is None or roc_auc is None:
        logger.warning(
            "Skipping ROC curve figure: ROC-AUC was not computable "
            "(ground truth had only one class)."
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"XGBoost (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ASSISTments External Validation\nROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"ROC curve figure saved to: {output_path}")


def save_metrics_csv(metrics: dict, output_path: Path) -> None:
    """Saves the scalar supervised-evaluation metrics as a one-row CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "dataset": "ASSISTments 2012-2013",
        "target_variable": "correct (aggregated to avg_score, thresholded)",
        "ground_truth_threshold": f"avg_score < {GROUND_TRUTH_AVG_SCORE_THRESHOLD}",
        "decision_threshold": DECISION_THRESHOLD,
        "n_records": metrics["n_records"],
        "n_positive": metrics["n_positive"],
        "n_negative": metrics["n_negative"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"] if metrics["roc_auc"] is not None else "N/A",
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)
    logger.info(f"Ground-truth metrics CSV saved to: {output_path}")


def write_markdown_report(metrics: dict, output_path: Path) -> None:
    """
    Writes the IEEE-paper-ready markdown report: dataset, target
    variable, evaluation metrics, interpretation, and limitations —
    per spec item 6. The Limitations section explicitly discloses the
    feature/label overlap described in the module docstring; this is
    not omitted or softened.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A (single-class ground truth)"
    cm = metrics["confusion_matrix"]

    report = f"""# ASSISTments External Validation — Ground-Truth Supervised Evaluation

## 1. Dataset

- **Source:** ASSISTments 2012-2013 (`2012-2013-data-with-predictions-4-final.csv`), preprocessed by the unmodified `preprocess_assistments.py` into `assistments_student_profiles.csv`.
- **Unit of analysis:** one row per (student_class_id, student_id), matching the granularity the model was designed to score.
- **Records evaluated:** {metrics['n_records']}
- **Ground-truth class balance:** {metrics['n_positive']} predicted-risk-positive ({metrics['positive_pct']:.1f}%) vs. {metrics['n_negative']} predicted-risk-negative ({metrics['negative_pct']:.1f}%) students (by ground-truth label, not model output).

## 2. Target Variable (Ground Truth)

The raw ASSISTments export contains a real, observed correctness column, `correct` (1 if a student answered a given problem correctly, 0 otherwise). This is genuine ground-truth data, not a label invented for this evaluation.

`correct` is row-level (per problem interaction); the EduGuard-AI risk model predicts at the (student_class_id, student_id) level. `preprocess_assistments.py` already aggregates `correct` into `avg_score = mean(correct) * 100` at exactly this granularity. This evaluation defines:

```
ground_truth_risk = 1  if avg_score < {GROUND_TRUTH_AVG_SCORE_THRESHOLD:.0f}
ground_truth_risk = 0  otherwise
```

The {GROUND_TRUTH_AVG_SCORE_THRESHOLD:.0f}-point threshold is not a new cutoff invented for this report — it is the same threshold `risk_service.py` already uses elsewhere in this project to flag "Academic performance weakening," reused here rather than introducing an independent decision boundary.

This is a deterministic transformation of real observed data. It is **not** a synthetic label, **not** weak supervision, and **not** fabricated.

## 3. Model

The frozen, OULAD-trained XGBoost psychological-risk classifier (`academic_risk_xgboost.pkl`) is applied as-is, with no retraining or fine-tuning on ASSISTments data, using the same seven fixed behavioural features as training time.

## 4. Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-score | {metrics['f1_score']:.4f} |
| ROC-AUC | {roc_auc_str} |

**Confusion Matrix** (rows = ground truth, columns = predicted; see also `figures/assistments_confusion_matrix.png`):

| | Predicted: Low Risk | Predicted: High Risk |
|---|---|---|
| **Actual: Low Risk** | {cm[0][0]} | {cm[0][1]} |
| **Actual: High Risk** | {cm[1][0]} | {cm[1][1]} |

**ROC Curve:** see `figures/assistments_roc_curve.png`.

## 5. Interpretation

These metrics indicate how well the OULAD-trained model's risk predictions agree with an ASSISTments-native, correctness-derived risk label, on a population and platform the model was never trained on. Reasonable agreement supports the model's behavioural-risk signal generalizing beyond OULAD; disagreement is equally informative and should be read alongside the distribution-only comparison already reported in `assistments_predictions.csv` / the existing prediction-summary log, not in isolation.

## 6. Limitations

- **Feature/label overlap.** `ground_truth_risk` is derived from `avg_score`, which is *also* one of the seven input features the model consumes. This is a materially different situation from the original OULAD evaluation, where the label (`final_result`) was an outcome column fully independent of the input features. Strong agreement here is partly expected by construction and should not be read as equivalent evidence to an independent held-out test.
- **No true withdrawal/failure outcome.** ASSISTments has no equivalent of OULAD's `final_result` (Withdrawn/Fail/Pass/Distinction). The threshold-based proxy above is the closest available real, observed analogue, not an equivalent construct.
- **Single fixed threshold.** The {GROUND_TRUTH_AVG_SCORE_THRESHOLD:.0f}-point cutoff is one reasonable, previously-used-elsewhere choice; results have not been evaluated across alternative thresholds.
- **Class imbalance sensitivity.** Precision/Recall/F1 can be sensitive to the ground-truth class balance reported in Section 1; interpret alongside the confusion matrix, not Accuracy alone.
- **No retraining.** The model is applied frozen; any ASSISTments-specific distribution shift (see the separate prediction-summary log) is not corrected for.

## 7. Threshold Sensitivity Note

The metrics above are computed at a single ground-truth threshold (`avg_score < {GROUND_TRUTH_AVG_SCORE_THRESHOLD:.1f}`, resulting in the {metrics['positive_pct']:.1f}% / {metrics['negative_pct']:.1f}% class split reported in Section 1). All reported metrics — Accuracy, Precision, Recall, F1, ROC-AUC, and the confusion matrix — are conditional on this specific cutoff, which is configured through a single constant (`GROUND_TRUTH_AVG_SCORE_THRESHOLD`) in `generate_external_predictions.py`. ROC-AUC is threshold-independent with respect to the *decision* threshold on predicted probability, but it is still computed against this one *ground-truth* threshold, so it does not by itself address this sensitivity. A different cutoff would shift the class balance and could shift Precision/Recall/F1 accordingly (Accuracy and ROC-AUC would likely be more stable, but this has not been verified here). Reporting results across a small sweep of alternative thresholds is a natural, low-cost follow-up left for future work, rather than something resolved in this evaluation.

## 8. Reproducibility

Generated by `generate_external_predictions.py`. Inputs: `assistments_student_profiles.csv` (from `preprocess_assistments.py`, unmodified) and `academic_risk_xgboost.pkl` (from `preprocess_oulad.py`, unmodified). No synthetic or weakly-supervised labels were introduced at any step.
"""

    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Markdown validation report saved to: {output_path}")


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    try:
        # ---- 1. Load reconstructed ASSISTments behavioural profiles ----
        profiles = load_student_profiles(INPUT_PATH)

        # ---- 2. Load the frozen OULAD-trained XGBoost model ----
        model = load_model(MODEL_PATH)

        # ---- 3. Verify required feature columns are present ----
        validate_feature_columns(profiles)

        # ---- 4-6. Generate predicted_probability and predicted_label ----
        predictions = generate_predictions(model, profiles)

        # ---- 7. Save output, preserving all original columns ----
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Predictions saved to: {OUTPUT_PATH}")

        # ---- 8. Print prediction summary statistics ----
        log_prediction_summary(predictions)

        # ---- 9. (REVIEWER EXTENSION) Derive real ground-truth labels ----
        labeled = derive_ground_truth_labels(predictions)

        # ---- 10. Compute supervised evaluation metrics ----
        metrics = compute_supervised_metrics(labeled)

        # ---- 11. Save metrics CSV ----
        save_metrics_csv(metrics, GROUND_TRUTH_METRICS_PATH)

        # ---- 12. Save confusion matrix + ROC curve figures ----
        plot_confusion_matrix(metrics["confusion_matrix"], CONFUSION_MATRIX_FIG_PATH)
        plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], ROC_CURVE_FIG_PATH)

        # ---- 13. Write the IEEE-ready markdown validation report ----
        write_markdown_report(metrics, MARKDOWN_REPORT_PATH)

        # ---- 14. Log a concise supervised-evaluation summary ----
        logger.info("=" * 60)
        logger.info("EduGuard-AI :: ASSISTments ground-truth evaluation summary")
        logger.info("=" * 60)
        logger.info(f"Ground-truth threshold:   avg_score < {GROUND_TRUTH_AVG_SCORE_THRESHOLD:.1f}")
        logger.info(
            f"Class distribution:       {metrics['n_positive']} High Risk "
            f"({metrics['positive_pct']:.1f}%), {metrics['n_negative']} Low Risk "
            f"({metrics['negative_pct']:.1f}%), n={metrics['n_records']}"
        )
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-score:  {metrics['f1_score']:.4f}")
        roc_auc_log = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
        logger.info(f"ROC-AUC:   {roc_auc_log}")
        logger.info("=" * 60)

    except (FileNotFoundError, ValueError) as exc:
        # Fail loudly and clearly rather than proceeding with invalid
        # or incomplete data / model state.
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
