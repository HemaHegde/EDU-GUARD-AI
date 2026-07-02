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

Run:
    python generate_external_predictions.py

Only the model/input paths below may need editing; everything else is
self-contained. Paths are resolved automatically relative to this
script's location, so the script works regardless of where the
repository is cloned.

Allowed dependencies: pandas, numpy, joblib, pathlib (standard library
`logging` is used for structured console output).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

    except (FileNotFoundError, ValueError) as exc:
        # Fail loudly and clearly rather than proceeding with invalid
        # or incomplete data / model state.
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
