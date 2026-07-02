"""
==================================================================================
ETHICS AND FAIRNESS EVALUATION
Academic Risk Prediction System (Frozen XGBoost Model)
==================================================================================

This standalone script performs a post-hoc ethics and fairness audit of the
FROZEN academic_risk_xgboost.pkl model against real student data.

STRICT CONSTRAINTS (enforced throughout this script):
    - The model is NEVER retrained. .fit() is never called on any estimator.
    - No backend / service files are read or modified.
    - No synthetic students are generated.
    - No missing values are imputed or estimated. Rows with missing values in
      required model features or in a given demographic column are excluded
      ONLY from the analyses that require that column, and this is logged.
    - Demographic groups are never fabricated. Only demographic columns that
      are actually present in the real dataset are analyzed.
    - Fairness metrics (statistical parity difference, disparate impact ratio)
      are computed only when the underlying group sizes/statistics make the
      computation mathematically valid (see METHODOLOGY notes in the report).

USAGE:
    python ethics_fairness.py

    No command-line arguments are required. The script auto-detects the
    project structure (dataset + frozen model) relative to its own location.

OUTPUTS (created relative to this script's directory):
    outputs/fairness_metrics.csv
    outputs/group_statistics.csv
    outputs/calibration_by_group.csv
    figures/demographic_distribution.png
    figures/fairness_comparison.png
    figures/prediction_rate_by_group.png
    figures/calibration_curves.png
    reports/fairness_report.md
==================================================================================
"""

import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ==================================================================================
# CONSTANTS
# ==================================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "academic_risk_xgboost.pkl"
DATASET_FILENAME = "final_student_psychology_dataset.csv"

# Feature order the model was trained on (see preprocess_oulad.py). This order
# MUST match training exactly since the model is used as-is (frozen, no refit).
MODEL_FEATURES = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
]

TRUE_LABEL_COLUMN = "psychological_risk"

# Candidate demographic / protected-attribute columns. Only columns that are
# actually present in the real dataset are evaluated; the rest are skipped
# and documented. "such as" in the task spec is non-exhaustive, so imd_band
# (a real socio-economic deprivation indicator present in OULAD-derived data)
# is included as an additional legitimate demographic attribute when present.
CANDIDATE_DEMOGRAPHIC_COLUMNS = [
    "gender",
    "age_band",
    "disability",
    "highest_education",
    "region",
    "imd_band",
]

RISK_THRESHOLD = 0.5          # decision threshold used by the frozen model's business logic
MIN_GROUP_SIZE = 30           # minimum sample size for a group to be considered
                               # statistically meaningful for fairness-ratio metrics
MIN_GROUP_SIZE_BASIC = 5      # minimum sample size to report basic descriptive stats at all
N_CALIBRATION_BINS = 5
DPI = 300

# Statistical rigor settings (reproducibility-oriented; does not alter any
# previously computed point-estimate metric or output column).
CI_LEVEL = 0.95
BOOTSTRAP_N_RESAMPLES = 1000  # number of bootstrap resamples for 95% CIs
BOOTSTRAP_SEED = 42           # fixed seed for reproducible bootstrap CIs
BOOTSTRAP_MIN_GROUP_SIZE = MIN_GROUP_SIZE  # only bootstrap groups meeting this size

# ==================================================================================
# LOGGING
# ==================================================================================

logger = logging.getLogger("ethics_fairness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
)
logger.handlers = [_handler]


# ==================================================================================
# PATH RESOLUTION (auto-detect project structure)
# ==================================================================================

def _find_file_upwards(filename, start_dir, candidate_relative_dirs, max_levels=5):
    """
    Search for `filename` starting from `start_dir`, first checking a list of
    candidate relative sub-paths at each ancestor level, then falling back to
    a bounded recursive search of each ancestor directory.
    """
    current = os.path.abspath(start_dir)

    # Pass 1: check explicit candidate relative paths at each ancestor level.
    for _ in range(max_levels + 1):
        for rel in candidate_relative_dirs:
            candidate = os.path.normpath(os.path.join(current, rel, filename))
            if os.path.isfile(candidate):
                return candidate
        current = os.path.dirname(current)

    # Pass 2: bounded recursive search of ancestor directories (handles
    # unconventional project layouts without scanning the whole filesystem).
    current = os.path.abspath(start_dir)
    for _ in range(max_levels + 1):
        for root, dirs, files in os.walk(current):
            # Avoid descending into virtual envs / node_modules / hidden dirs.
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       ("node_modules", "venv", ".venv", "__pycache__")]
            if filename in files:
                return os.path.join(root, filename)
            # Keep the search shallow-ish per level to stay fast.
        current = os.path.dirname(current)

    return None


def resolve_model_path():
    override = os.environ.get("ACADEMIC_RISK_MODEL_PATH")
    if override and os.path.isfile(override):
        return override

    candidates = [
        "..",                 # ml/academic_risk_xgboost.pkl
        ".",                  # same directory
        "../..",              # project_root/academic_risk_xgboost.pkl
        "../../ml",
    ]
    path = _find_file_upwards(MODEL_FILENAME, SCRIPT_DIR, candidates)
    return path


def resolve_dataset_path():
    override = os.environ.get("ACADEMIC_RISK_DATASET_PATH")
    if override and os.path.isfile(override):
        return override

    candidates = [
        "../../datasets",     # project_root/datasets/final_student_psychology_dataset.csv
        "../datasets",
        "../../data",
        "../data",
        ".",
    ]
    path = _find_file_upwards(DATASET_FILENAME, SCRIPT_DIR, candidates)
    return path


# ==================================================================================
# OUTPUT DIRECTORIES
# ==================================================================================

def ensure_output_dirs():
    dirs = {
        "outputs": os.path.join(SCRIPT_DIR, "outputs"),
        "figures": os.path.join(SCRIPT_DIR, "figures"),
        "reports": os.path.join(SCRIPT_DIR, "reports"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ==================================================================================
# DATA / MODEL LOADING
# ==================================================================================

def load_model(model_path):
    logger.info(f"Loading FROZEN model from: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully. Type: {type(model).__name__}")
    if hasattr(model, "fit"):
        logger.info("Confirmed model exposes sklearn-style API. "
                     "This script will NOT call .fit() at any point.")
    return model


def resolve_model_features(model):
    """
    Determine the input feature order the frozen model expects.

    Preference order (does not change any evaluation logic — only decides
    which columns are fed into the already-frozen model, exactly as before):
        1. `model.feature_names_in_` (sklearn-style attribute, populated when
           the estimator was fit on a DataFrame with named columns).
        2. `model.get_booster().feature_names` (XGBoost-specific fallback).
        3. The hardcoded MODEL_FEATURES list (original behavior), used only
           if neither of the above is available.

    This makes the script resilient to future preprocessing changes without
    silently guessing at feature values.
    """
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        names = list(model.feature_names_in_)
        if names and all(isinstance(n, str) for n in names):
            logger.info(f"Using model.feature_names_in_ for input feature order: {names}")
            return names

    try:
        booster = model.get_booster()
        names = booster.feature_names
        if names:
            logger.info(f"Using XGBoost booster.feature_names for input feature order: {names}")
            return list(names)
    except Exception:
        pass

    logger.info(
        "Model does not expose feature_names_in_ or booster.feature_names; "
        f"falling back to hardcoded MODEL_FEATURES: {MODEL_FEATURES}"
    )
    return list(MODEL_FEATURES)


def load_dataset(dataset_path):
    logger.info(f"Loading real dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    return df


def validate_required_columns(df, feature_columns):
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataset is missing required model feature columns: {missing_features}. "
            "Cannot proceed without fabricating values, which is prohibited."
        )
    if TRUE_LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset is missing the true label column '{TRUE_LABEL_COLUMN}'. "
            "Cannot proceed without fabricating values, which is prohibited."
        )
    logger.info("All required model feature columns and the true label column are present.")


def detect_demographic_columns(df):
    present = []
    skipped = []
    for col in CANDIDATE_DEMOGRAPHIC_COLUMNS:
        if col in df.columns:
            present.append(col)
        else:
            skipped.append(col)
    logger.info(f"Demographic columns detected in dataset: {present}")
    if skipped:
        logger.info(f"Demographic columns NOT present (skipped gracefully): {skipped}")
    return present, skipped


# ==================================================================================
# INFERENCE (frozen model, no retraining)
# ==================================================================================

def run_inference(model, df, feature_columns):
    logger.info("Running inference with the frozen model (no fitting performed)...")

    # Only rows with complete, real (non-missing, non-estimated) feature values
    # and a real true label are scored. No values are imputed.
    feature_df = df[feature_columns]
    valid_mask = feature_df.notna().all(axis=1) & df[TRUE_LABEL_COLUMN].notna()

    n_total = len(df)
    n_valid = int(valid_mask.sum())
    n_excluded = n_total - n_valid

    if n_excluded > 0:
        logger.info(
            f"{n_excluded} of {n_total} rows excluded from inference due to missing "
            f"required feature/label values (no imputation performed, per constraints)."
        )
    else:
        logger.info(f"All {n_total} rows have complete required feature/label values.")

    X = feature_df.loc[valid_mask, feature_columns]
    y_true = df.loc[valid_mask, TRUE_LABEL_COLUMN].astype(int)

    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= RISK_THRESHOLD).astype(int)

    result = df.loc[valid_mask].copy()
    result["predicted_probability"] = proba
    result["predicted_label"] = y_pred
    result["true_label"] = y_true.values

    logger.info(f"Inference complete on {len(result)} students.")
    return result


# ==================================================================================
# GROUP-WISE METRICS
# ==================================================================================

def bootstrap_group_metrics(y_true, y_pred, n_boot=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """
    Compute 95% bootstrap confidence intervals (percentile method) for
    accuracy, precision, recall, F1, and positive-prediction-rate for a
    single group, via vectorized resampling with replacement.

    Returns a dict mapping metric name -> (ci_lower, ci_upper), or None if
    the group is too small / degenerate for a meaningful bootstrap.
    This is purely descriptive uncertainty quantification around the
    already-computed point estimates; it does not change those point
    estimates.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    n = len(y_true)

    if n < BOOTSTRAP_MIN_GROUP_SIZE:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)

    yt = y_true[idx]
    yp = y_pred[idx]

    tp = np.sum((yt == 1) & (yp == 1), axis=1).astype(np.float64)
    fp = np.sum((yt == 0) & (yp == 1), axis=1).astype(np.float64)
    fn = np.sum((yt == 1) & (yp == 0), axis=1).astype(np.float64)
    tn = np.sum((yt == 0) & (yp == 0), axis=1).astype(np.float64)

    accuracy_b = (tp + tn) / n

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_b = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        recall_b = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        f1_b = np.where(
            (precision_b + recall_b) > 0,
            2 * precision_b * recall_b / (precision_b + recall_b),
            np.nan,
        )

    ppr_b = (tp + fp) / n

    alpha = 1 - CI_LEVEL
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    def _ci(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) < max(10, int(0.5 * n_boot)):
            return (np.nan, np.nan)
        return (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))

    return {
        "accuracy": _ci(accuracy_b),
        "precision": _ci(precision_b),
        "recall": _ci(recall_b),
        "f1_score": _ci(f1_b),
        "positive_prediction_rate": _ci(ppr_b),
    }


def bootstrap_spd_ci(group_pred, reference_pred, n_boot=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """
    95% bootstrap CI (percentile method) for the Statistical Parity
    Difference between a group's positive-prediction rate and the
    reference group's positive-prediction rate, via independent resampling
    of each group's predicted labels. Does not change the point-estimate
    SPD already computed elsewhere in the script.
    """
    group_pred = np.asarray(group_pred, dtype=np.int8)
    reference_pred = np.asarray(reference_pred, dtype=np.int8)
    n_g, n_r = len(group_pred), len(reference_pred)

    if n_g < BOOTSTRAP_MIN_GROUP_SIZE or n_r < BOOTSTRAP_MIN_GROUP_SIZE:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    idx_g = rng.integers(0, n_g, size=(n_boot, n_g), dtype=np.int64)
    idx_r = rng.integers(0, n_r, size=(n_boot, n_r), dtype=np.int64)

    ppr_g = group_pred[idx_g].mean(axis=1)
    ppr_r = reference_pred[idx_r].mean(axis=1)
    spd_b = ppr_g - ppr_r

    alpha = 1 - CI_LEVEL
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return (float(np.percentile(spd_b, lo_q)), float(np.percentile(spd_b, hi_q)))


def cohens_h(p1, p2):
    """
    Cohen's h effect size for the difference between two proportions.
    Standard interpretation (Cohen, 1988): |h| < 0.2 negligible,
    0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
    Returns (h, magnitude_label) or (nan, "not computable") if either
    proportion is undefined.
    """
    if p1 is None or p2 is None or (isinstance(p1, float) and np.isnan(p1)) or \
            (isinstance(p2, float) and np.isnan(p2)):
        return (np.nan, "not computable")

    h = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1))) - 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))
    abs_h = abs(h)
    if abs_h < 0.2:
        label = "negligible"
    elif abs_h < 0.5:
        label = "small"
    elif abs_h < 0.8:
        label = "medium"
    else:
        label = "large"
    return (float(h), label)


def safe_metric(func, y_true, y_pred, **kwargs):
    try:
        return func(y_true, y_pred, zero_division=0, **kwargs)
    except TypeError:
        # accuracy_score does not accept zero_division
        return func(y_true, y_pred)
    except Exception:
        return np.nan


def compute_group_statistics(result, demographic_columns):
    """
    For every real demographic column present, compute per-group descriptive
    and performance metrics. Groups below MIN_GROUP_SIZE_BASIC are reported
    with counts only (metrics left as NaN) since metrics would not be
    statistically meaningful.
    """
    rows = []

    for col in demographic_columns:
        sub = result[[col, "true_label", "predicted_label", "predicted_probability"]].dropna(
            subset=[col]
        )
        n_excluded = len(result) - len(sub)
        if n_excluded > 0:
            logger.info(
                f"'{col}': {n_excluded} rows excluded from this attribute's analysis "
                f"due to missing values in '{col}' (no estimation performed)."
            )

        groups = sub[col].value_counts()
        logger.info(f"Attribute '{col}' has {len(groups)} real observed group(s): "
                     f"{groups.to_dict()}")

        for group_value, count in groups.items():
            grp = sub[sub[col] == group_value]

            row = {
                "attribute": col,
                "group": group_value,
                "sample_count": int(count),
                "positive_prediction_rate": np.nan,
                "avg_predicted_probability": np.nan,
                "avg_true_label": np.nan,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": np.nan,
                "accuracy_ci_lower": np.nan,
                "accuracy_ci_upper": np.nan,
                "recall_ci_lower": np.nan,
                "recall_ci_upper": np.nan,
                "precision_ci_lower": np.nan,
                "precision_ci_upper": np.nan,
                "f1_score_ci_lower": np.nan,
                "f1_score_ci_upper": np.nan,
                "positive_prediction_rate_ci_lower": np.nan,
                "positive_prediction_rate_ci_upper": np.nan,
                "note": "",
            }

            if count < MIN_GROUP_SIZE_BASIC:
                row["note"] = (
                    f"Group size ({count}) below minimum threshold "
                    f"({MIN_GROUP_SIZE_BASIC}) for reliable metric estimation; "
                    f"only sample count reported."
                )
                rows.append(row)
                continue

            y_true = grp["true_label"].astype(int)
            y_pred = grp["predicted_label"].astype(int)
            proba = grp["predicted_probability"].astype(float)

            row["positive_prediction_rate"] = float(y_pred.mean())
            row["avg_predicted_probability"] = float(proba.mean())
            row["avg_true_label"] = float(y_true.mean())
            row["accuracy"] = float(accuracy_score(y_true, y_pred))
            row["precision"] = float(safe_metric(precision_score, y_true, y_pred))
            row["recall"] = float(safe_metric(recall_score, y_true, y_pred))
            row["f1_score"] = float(safe_metric(f1_score, y_true, y_pred))

            notes = []
            if y_true.nunique() < 2:
                notes.append(
                    "Only one true-label class present in this group; "
                    "precision/recall/F1 computed w.r.t. the single class "
                    "observed and may be degenerate."
                )

            # 95% bootstrap confidence intervals (percentile method) for the
            # major performance metrics. Only computed for groups meeting
            # BOOTSTRAP_MIN_GROUP_SIZE; point estimates above are unaffected.
            if count >= BOOTSTRAP_MIN_GROUP_SIZE:
                ci = bootstrap_group_metrics(y_true.values, y_pred.values)
                if ci is not None:
                    row["accuracy_ci_lower"], row["accuracy_ci_upper"] = ci["accuracy"]
                    row["precision_ci_lower"], row["precision_ci_upper"] = ci["precision"]
                    row["recall_ci_lower"], row["recall_ci_upper"] = ci["recall"]
                    row["f1_score_ci_lower"], row["f1_score_ci_upper"] = ci["f1_score"]
                    (row["positive_prediction_rate_ci_lower"],
                     row["positive_prediction_rate_ci_upper"]) = ci["positive_prediction_rate"]
            else:
                notes.append(
                    f"Group size ({count}) below bootstrap CI threshold "
                    f"({BOOTSTRAP_MIN_GROUP_SIZE}); confidence intervals not computed."
                )

            row["note"] = " ".join(notes)
            rows.append(row)

    group_stats_df = pd.DataFrame(rows)
    return group_stats_df


# ==================================================================================
# FAIRNESS METRICS (statistical parity difference, disparate impact ratio)
# ==================================================================================

def _group_tpr_fpr(true_label, predicted_label):
    """
    Compute True Positive Rate (recall on the positive class) and False
    Positive Rate for a single group's raw labels. Returns (tpr, fpr),
    each NaN if not mathematically defined (no positives for TPR, no
    negatives for FPR) rather than estimated.
    """
    y_true = np.asarray(true_label, dtype=np.int8)
    y_pred = np.asarray(predicted_label, dtype=np.int8)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    tpr = float(y_pred[pos_mask].mean()) if pos_mask.sum() > 0 else np.nan
    fpr = float(y_pred[neg_mask].mean()) if neg_mask.sum() > 0 else np.nan
    return tpr, fpr


def compute_fairness_metrics(group_stats_df, demographic_columns, result):
    """
    For each demographic attribute with 2+ statistically meaningful groups
    (sample_count >= MIN_GROUP_SIZE and a valid positive_prediction_rate),
    compute:
        - Statistical Parity Difference (SPD) = P(pred=1|group) - P(pred=1|reference)
        - Disparate Impact Ratio (DIR)        = P(pred=1|group) / P(pred=1|reference)
        - Equal Opportunity Difference (EOD)  = TPR(group) - TPR(reference)
        - False Positive Rate Difference (FPRD) = FPR(group) - FPR(reference)
        - Cohen's h effect size for the SPD proportion comparison
        - 95% bootstrap CI (percentile method) for SPD

    The reference group for each attribute is the group with the largest
    valid sample size (the most-represented real group in the data), which
    is a standard, non-fabricated choice grounded entirely in observed data.
    Every metric above is computed only when mathematically valid for the
    groups involved (see per-metric checks below); otherwise it is left as
    NaN and the reason is recorded in `notes`.
    """
    rows = []

    for col in demographic_columns:
        attr_df = group_stats_df[
            (group_stats_df["attribute"] == col)
            & (group_stats_df["sample_count"] >= MIN_GROUP_SIZE)
            & (group_stats_df["positive_prediction_rate"].notna())
        ].copy()

        if len(attr_df) < 2:
            logger.info(
                f"'{col}': fewer than 2 groups meet the minimum size threshold "
                f"({MIN_GROUP_SIZE}) with valid metrics; SPD/DIR not computed "
                f"for this attribute."
            )
            continue

        reference_row = attr_df.sort_values("sample_count", ascending=False).iloc[0]
        reference_group = reference_row["group"]
        reference_rate = reference_row["positive_prediction_rate"]

        ref_sub = result[result[col] == reference_group]
        reference_pred_arr = ref_sub["predicted_label"].astype(int).values
        reference_tpr, reference_fpr = _group_tpr_fpr(
            ref_sub["true_label"].astype(int).values, reference_pred_arr
        )

        for _, r in attr_df.iterrows():
            if r["group"] == reference_group:
                continue

            notes = []

            spd = r["positive_prediction_rate"] - reference_rate

            if reference_rate > 0:
                dir_ratio = r["positive_prediction_rate"] / reference_rate
            else:
                dir_ratio = np.nan
                notes.append("DIR not computed: reference group positive prediction rate is 0.")
                logger.info(
                    f"'{col}' group '{r['group']}': disparate impact ratio not "
                    f"computed (reference group positive prediction rate is 0)."
                )

            # ---- Equal Opportunity Difference / False Positive Rate Difference ----
            grp_sub = result[result[col] == r["group"]]
            grp_pred_arr = grp_sub["predicted_label"].astype(int).values
            grp_tpr, grp_fpr = _group_tpr_fpr(
                grp_sub["true_label"].astype(int).values, grp_pred_arr
            )

            if not np.isnan(grp_tpr) and not np.isnan(reference_tpr):
                eod = grp_tpr - reference_tpr
            else:
                eod = np.nan
                notes.append(
                    "EOD not computed: group and/or reference group has no positive "
                    "true-label cases, so TPR is undefined."
                )

            if not np.isnan(grp_fpr) and not np.isnan(reference_fpr):
                fprd = grp_fpr - reference_fpr
            else:
                fprd = np.nan
                notes.append(
                    "FPRD not computed: group and/or reference group has no negative "
                    "true-label cases, so FPR is undefined."
                )

            # ---- Effect size (Cohen's h) for the SPD proportion comparison ----
            h_value, h_magnitude = cohens_h(r["positive_prediction_rate"], reference_rate)

            # ---- 95% bootstrap CI for SPD ----
            spd_ci_lower, spd_ci_upper = bootstrap_spd_ci(grp_pred_arr, reference_pred_arr)
            if np.isnan(spd_ci_lower):
                notes.append(
                    f"SPD bootstrap CI not computed: group or reference group below "
                    f"minimum size ({BOOTSTRAP_MIN_GROUP_SIZE})."
                )

            rows.append({
                "attribute": col,
                "reference_group": reference_group,
                "reference_group_size": int(reference_row["sample_count"]),
                "reference_positive_rate": reference_rate,
                "reference_tpr": reference_tpr,
                "reference_fpr": reference_fpr,
                "compared_group": r["group"],
                "compared_group_size": int(r["sample_count"]),
                "compared_positive_rate": r["positive_prediction_rate"],
                "compared_tpr": grp_tpr,
                "compared_fpr": grp_fpr,
                "statistical_parity_difference": spd,
                "statistical_parity_difference_pct_points": spd * 100,
                "spd_ci_lower": spd_ci_lower,
                "spd_ci_upper": spd_ci_upper,
                "disparate_impact_ratio": dir_ratio,
                "equal_opportunity_difference": eod,
                "false_positive_rate_difference": fprd,
                "cohens_h": h_value,
                "effect_size_magnitude": h_magnitude,
                "four_fifths_rule_flag": bool(
                    not np.isnan(dir_ratio) and (dir_ratio < 0.8 or dir_ratio > 1.25)
                ) if not np.isnan(dir_ratio) else False,
                "notes": " ".join(notes),
            })

    fairness_df = pd.DataFrame(rows)
    return fairness_df


# ==================================================================================
# CALIBRATION BY GROUP
# ==================================================================================

def compute_calibration_by_group(result, demographic_columns, n_bins=N_CALIBRATION_BINS):
    """
    For each demographic group, bin students by predicted probability and
    compare mean predicted probability to observed positive rate (mean true
    label) in that bin. This reveals whether the model is equally
    well-calibrated across groups (a core fairness property distinct from
    parity of outcomes).
    """
    rows = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for col in demographic_columns:
        sub = result[[col, "true_label", "predicted_probability"]].dropna(subset=[col])
        for group_value, grp in sub.groupby(col):
            if len(grp) < MIN_GROUP_SIZE_BASIC:
                continue

            bin_idx = np.digitize(grp["predicted_probability"], bin_edges[1:-1], right=True)
            for b in range(n_bins):
                bin_mask = bin_idx == b
                bin_n = int(bin_mask.sum())
                if bin_n == 0:
                    continue
                rows.append({
                    "attribute": col,
                    "group": group_value,
                    "probability_bin_low": round(bin_edges[b], 2),
                    "probability_bin_high": round(bin_edges[b + 1], 2),
                    "bin_sample_count": bin_n,
                    "mean_predicted_probability": float(
                        grp.loc[bin_mask, "predicted_probability"].mean()
                    ),
                    "observed_positive_rate": float(
                        grp.loc[bin_mask, "true_label"].mean()
                    ),
                })

    calibration_df = pd.DataFrame(rows)
    return calibration_df


# ==================================================================================
# FIGURES
# ==================================================================================

PALETTE = ["#2E5266", "#6E8898", "#9FB8AD", "#D7B377", "#B85042", "#5A7D7C", "#8C6A5D", "#3E4A61"]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def fig_demographic_distribution(result, demographic_columns, out_path):
    if not demographic_columns:
        logger.info("No demographic columns available; skipping demographic_distribution.png")
        return

    n = len(demographic_columns)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(demographic_columns):
        ax = axes[i]
        counts = result[col].dropna().value_counts().sort_values(ascending=True)
        colors = [PALETTE[j % len(PALETTE)] for j in range(len(counts))]
        ax.barh(counts.index.astype(str), counts.values, color=colors, edgecolor="white")
        ax.set_title(f"Distribution by {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Student count")
        style_axes(ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Dataset Demographic Distribution (Real Students Only)",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def fig_prediction_rate_by_group(group_stats_df, out_path):
    plot_df = group_stats_df.dropna(subset=["positive_prediction_rate"])
    if plot_df.empty:
        logger.info("No valid group metrics; skipping prediction_rate_by_group.png")
        return

    attrs = plot_df["attribute"].unique()
    n = len(attrs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, attr in enumerate(attrs):
        ax = axes[i]
        sub = plot_df[plot_df["attribute"] == attr].sort_values("positive_prediction_rate")
        colors = [PALETTE[j % len(PALETTE)] for j in range(len(sub))]
        ax.barh(sub["group"].astype(str), sub["positive_prediction_rate"] * 100,
                color=colors, edgecolor="white")
        ax.set_title(f"Positive Prediction Rate — {attr}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Positive prediction rate (%)")
        style_axes(ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Model Positive-Prediction Rate by Demographic Group",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def fig_fairness_comparison(fairness_df, out_path):
    if fairness_df.empty:
        logger.info("No fairness metrics computed; skipping fairness_comparison.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.4 * len(fairness_df) + 2)))

    labels = fairness_df["attribute"] + ": " + fairness_df["compared_group"].astype(str)

    ax = axes[0]
    colors = ["#B85042" if abs(v) > 0.1 else "#2E5266"
              for v in fairness_df["statistical_parity_difference"]]
    ax.barh(labels, fairness_df["statistical_parity_difference"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Statistical Parity Difference\n(vs. largest group per attribute)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("SPD")
    style_axes(ax)

    ax = axes[1]
    dir_vals = fairness_df["disparate_impact_ratio"]
    colors = ["#B85042" if (pd.notna(v) and (v < 0.8 or v > 1.25)) else "#2E5266"
              for v in dir_vals]
    ax.barh(labels, dir_vals.fillna(0), color=colors, edgecolor="white")
    ax.axvline(1.0, color="black", linewidth=0.8, label="Parity (1.0)")
    ax.axvline(0.8, color="gray", linestyle="--", linewidth=0.8, label="80% rule bounds")
    ax.axvline(1.25, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Disparate Impact Ratio\n(vs. largest group per attribute)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("DIR")
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)

    fig.suptitle("Fairness Metrics by Demographic Attribute", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def fig_calibration_curves(calibration_df, out_path):
    if calibration_df.empty:
        logger.info("No calibration data computed; skipping calibration_curves.png")
        return

    attrs = calibration_df["attribute"].unique()
    n = len(attrs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, attr in enumerate(attrs):
        ax = axes[i]
        sub = calibration_df[calibration_df["attribute"] == attr]
        for j, (group_value, grp) in enumerate(sub.groupby("group")):
            grp = grp.sort_values("mean_predicted_probability")
            ax.plot(grp["mean_predicted_probability"], grp["observed_positive_rate"],
                    marker="o", label=str(group_value), color=PALETTE[j % len(PALETTE)],
                    linewidth=1.6, markersize=4)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8,
                label="Perfect calibration")
        ax.set_title(f"Calibration — {attr}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, loc="best")
        style_axes(ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Calibration Curves by Demographic Group", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


# ==================================================================================
# REPORT GENERATION
# ==================================================================================

def build_report(
    report_path,
    df,
    result,
    demographic_columns,
    skipped_columns,
    group_stats_df,
    fairness_df,
    calibration_df,
    model_path,
    dataset_path,
    feature_columns,
):
    n_total = len(df)
    n_scored = len(result)
    n_excluded = n_total - n_scored

    overall_positive_rate = float(result["predicted_label"].mean()) if n_scored else float("nan")
    overall_true_rate = float(result["true_label"].mean()) if n_scored else float("nan")
    overall_accuracy = float(accuracy_score(result["true_label"], result["predicted_label"])) \
        if n_scored else float("nan")

    n_groups_analyzed = int(group_stats_df.shape[0]) if not group_stats_df.empty else 0
    n_fairness_flags = int(fairness_df["four_fifths_rule_flag"].sum()) if not fairness_df.empty else 0
    n_comparisons = int(fairness_df.shape[0]) if not fairness_df.empty else 0

    lines = []
    lines.append("# Ethics and Fairness Evaluation Report")
    lines.append("## Academic Risk Prediction System (Frozen XGBoost Model)")
    lines.append("")
    lines.append(f"*Generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"by `ethics_fairness.py`. All figures and tables in this report are "
                  f"produced directly from real, observed data — no values in this report "
                  f"are estimated, simulated, or fabricated.*")
    lines.append("")

    # ---------------- Executive Summary ----------------
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("**Key Findings**")
    lines.append("")
    lines.append(f"- {len(demographic_columns)} real demographic attribute(s) evaluated "
                 f"(`{', '.join(demographic_columns) if demographic_columns else 'none'}`)")
    lines.append(f"- {n_groups_analyzed} demographic group(s) analyzed across those attributes")
    lines.append(f"- {n_scored} of {n_total} students scored by the frozen model "
                 f"({n_excluded} excluded due to missing values, no imputation performed)")
    lines.append(f"- {n_comparisons} group-vs-reference fairness comparison(s) computed")
    lines.append(f"- {n_fairness_flags} four-fifths-rule fairness flag(s) detected")
    lines.append(f"- Overall predicted positive rate: {overall_positive_rate:.3f} | "
                 f"Overall observed true positive rate: {overall_true_rate:.3f} | "
                 f"Overall accuracy: {overall_accuracy:.3f}")
    lines.append("- No retraining performed — the frozen `.pkl` model was used strictly in "
                 "inference mode (`predict_proba` only, `.fit()` never called)")
    lines.append(f"- Model input features used (auto-detected from the model where available): "
                 f"`{', '.join(feature_columns)}`")
    lines.append("")

    # ---------------- 1. Objective ----------------
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "This report presents a post-hoc ethics and fairness audit of the frozen "
        "academic risk prediction model (`academic_risk_xgboost.pkl`). The objective "
        "is to determine whether the model's predictions differ systematically across "
        "real demographic subgroups present in the student population, and whether "
        "such differences raise fairness concerns that warrant mitigation before the "
        "model is used to inform student support interventions. The model itself is "
        "not modified, retrained, or fine-tuned as part of this evaluation; it is "
        "used strictly in inference mode against the frozen artifact provided."
    )
    lines.append("")

    # ---------------- 2. Methodology ----------------
    lines.append("## 2. Methodology")
    lines.append("")
    lines.append(f"- **Model artifact:** `{os.path.relpath(model_path, SCRIPT_DIR)}` "
                 f"(loaded via `joblib.load`; `.fit()` is never called).")
    lines.append(f"- **Dataset:** `{os.path.relpath(dataset_path, SCRIPT_DIR)}` "
                 f"({n_total} real student records).")
    lines.append(f"- **Model input features (order as reported by the frozen model itself "
                 f"via `feature_names_in_`/booster metadata when available, otherwise the "
                 f"hardcoded fallback list):** `{', '.join(feature_columns)}`.")
    lines.append(f"- **True label:** `{TRUE_LABEL_COLUMN}` (1 = Withdrawn/Fail, 0 = otherwise).")
    lines.append(f"- **Decision threshold:** predicted probability ≥ {RISK_THRESHOLD} → "
                 f"positive (\"at risk\") prediction, consistent with the deployed service.")
    lines.append(f"- **Records scored:** {n_scored} of {n_total} "
                 f"({n_excluded} excluded due to missing required feature/label values; "
                 f"no imputation was performed on excluded rows).")
    lines.append(
        "- **Demographic attribute detection:** the script scans the dataset for a "
        "candidate list of real demographic/protected attributes "
        f"(`{', '.join(CANDIDATE_DEMOGRAPHIC_COLUMNS)}`) and evaluates only those "
        "columns that actually exist in the data. No group is constructed or inferred; "
        "groups are exactly the distinct real values observed for each attribute."
    )
    lines.append(
        f"- **Minimum group size for descriptive metrics:** {MIN_GROUP_SIZE_BASIC} students. "
        f"Groups below this size are reported by sample count only."
    )
    lines.append(
        f"- **Minimum group size for fairness-ratio metrics (SPD / DIR):** {MIN_GROUP_SIZE} "
        "students in both the group and the reference group. This threshold guards "
        "against unstable ratios computed on small samples."
    )
    lines.append(
        "- **Reference group:** for each demographic attribute, the largest real group "
        "observed in the data is used as the reference for Statistical Parity "
        "Difference (SPD) and Disparate Impact Ratio (DIR). This is a data-driven "
        "choice, not an assumption about which group is 'privileged'; readers with "
        "domain-specific reference-group requirements should recompute against their "
        "chosen reference using the exported `group_statistics.csv`."
    )
    lines.append(
        "- **Calibration:** predicted probabilities are binned into "
        f"{N_CALIBRATION_BINS} equal-width bins per group; within each bin, the mean "
        "predicted probability is compared to the observed (true) positive rate."
    )
    lines.append(
        "- **Equal Opportunity Difference (EOD) and False Positive Rate Difference "
        "(FPRD):** computed as the difference between a group's True/False Positive "
        "Rate and the reference group's, respectively. EOD requires positive "
        "true-label cases in both groups (TPR is undefined otherwise); FPRD requires "
        "negative true-label cases in both groups (FPR is undefined otherwise). Where "
        "either precondition fails, the metric is left blank and the reason is stated "
        "in the comparison's notes, per the constraint against estimating missing values."
    )
    lines.append(
        "- **Effect size:** Cohen's h is reported alongside SPD/DIR for each group "
        "comparison, using the standard proportion-difference formulation "
        "(2·arcsin(√p₁) − 2·arcsin(√p₂)) with conventional magnitude bands "
        "(negligible < 0.2, small < 0.5, medium < 0.8, large ≥ 0.8)."
    )
    lines.append(
        f"- **Uncertainty quantification:** {CI_LEVEL:.0%} confidence intervals are "
        f"computed via percentile bootstrap ({BOOTSTRAP_N_RESAMPLES} resamples, fixed "
        f"seed {BOOTSTRAP_SEED} for reproducibility) for accuracy, precision, recall, "
        f"F1, and positive-prediction-rate at the group level, and for SPD at the "
        f"group-vs-reference comparison level. Bootstrap CIs are computed only for "
        f"groups with at least {BOOTSTRAP_MIN_GROUP_SIZE} students; smaller groups "
        f"report point estimates only, with this noted explicitly."
    )
    lines.append("")

    # ---------------- 3. Dataset demographics ----------------
    lines.append("## 3. Dataset Demographics")
    lines.append("")
    if demographic_columns:
        lines.append("The following real demographic attributes were detected and evaluated:")
        lines.append("")
        for col in demographic_columns:
            vc = df[col].dropna().value_counts()
            lines.append(f"- **{col}** ({df[col].notna().sum()} non-missing records, "
                         f"{len(vc)} distinct group(s)): "
                         + ", ".join(f"{idx} (n={cnt})" for idx, cnt in vc.items()))
    else:
        lines.append("No candidate demographic columns were found in the dataset.")
    lines.append("")

    if skipped_columns:
        lines.append(f"The following candidate demographic attributes were **not present** "
                     f"in the dataset and were skipped: `{', '.join(skipped_columns)}`.")
    else:
        lines.append("All candidate demographic attributes were present in the dataset.")
    lines.append("")
    lines.append("![Demographic Distribution](../figures/demographic_distribution.png)")
    lines.append("")

    # ---------------- 4. Group-wise evaluation ----------------
    lines.append("## 4. Group-wise Evaluation")
    lines.append("")
    lines.append(f"Overall (all scored students, n={n_scored}): predicted positive rate = "
                 f"{overall_positive_rate:.3f}, observed true positive rate = "
                 f"{overall_true_rate:.3f}, accuracy = {overall_accuracy:.3f}.")
    lines.append("")
    lines.append("Full per-group metrics, including 95% bootstrap confidence intervals, are "
                 "provided in `outputs/group_statistics.csv`. Summary below (accuracy and "
                 f"recall shown with their {CI_LEVEL:.0%} bootstrap CI in brackets where "
                 f"computed, i.e. groups with n ≥ {BOOTSTRAP_MIN_GROUP_SIZE}):")
    lines.append("")

    if not group_stats_df.empty:
        lines.append("| Attribute | Group | N | Pos. Pred. Rate | Avg Pred. Prob. | "
                     "Avg True Label | Accuracy [95% CI] | Precision | Recall [95% CI] | F1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in group_stats_df.iterrows():
            def fmt(v):
                return "n/a" if pd.isna(v) else f"{v:.3f}"
            def fmt_ci(point, lo, hi):
                if pd.isna(point):
                    return "n/a"
                if pd.isna(lo) or pd.isna(hi):
                    return f"{point:.3f}"
                return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
            lines.append(
                f"| {r['attribute']} | {r['group']} | {int(r['sample_count'])} | "
                f"{fmt(r['positive_prediction_rate'])} | {fmt(r['avg_predicted_probability'])} | "
                f"{fmt(r['avg_true_label'])} | "
                f"{fmt_ci(r['accuracy'], r.get('accuracy_ci_lower'), r.get('accuracy_ci_upper'))} | "
                f"{fmt(r['precision'])} | "
                f"{fmt_ci(r['recall'], r.get('recall_ci_lower'), r.get('recall_ci_upper'))} | "
                f"{fmt(r['f1_score'])} |"
            )
    else:
        lines.append("No group-wise metrics could be computed (no demographic columns present "
                     "or all groups below the minimum reporting threshold).")
    lines.append("")
    lines.append("![Prediction Rate by Group](../figures/prediction_rate_by_group.png)")
    lines.append("")
    lines.append("![Calibration Curves](../figures/calibration_curves.png)")
    lines.append("")

    # ---------------- 5. Fairness observations ----------------
    lines.append("## 5. Fairness Observations")
    lines.append("")
    if not fairness_df.empty:
        lines.append(
            "Statistical Parity Difference (SPD), Disparate Impact Ratio (DIR), Equal "
            "Opportunity Difference (EOD), and False Positive Rate Difference (FPRD) were "
            "computed for demographic attributes with at least two groups meeting the "
            f"minimum sample size ({MIN_GROUP_SIZE}). EOD and FPRD are computed only when "
            "the group and reference group both contain the relevant true-label class "
            "(positives for EOD, negatives for FPRD); where that precondition fails the "
            "cell is marked n/a and the reason is recorded in `outputs/fairness_metrics.csv`. "
            "SPD is additionally reported with its 95% bootstrap confidence interval and "
            "with Cohen's h as a standardized effect size. The U.S. EEOC \"four-fifths rule\" "
            "(DIR outside [0.8, 1.25]) is used purely as a descriptive flag, not as a "
            "legal or normative conclusion."
        )
        lines.append("")
        lines.append("| Attribute | Reference (n) | Compared (n) | Ref. Pos. Rate | "
                     "Compared Pos. Rate | SPD (95% CI) | Effect size (Cohen's h) | DIR | "
                     "EOD | FPRD | Four-Fifths Flag |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in fairness_df.iterrows():
            def fmt3(v):
                return "n/a" if pd.isna(v) else f"{v:.3f}"
            dir_str = fmt3(r["disparate_impact_ratio"])
            eod_str = fmt3(r["equal_opportunity_difference"])
            fprd_str = fmt3(r["false_positive_rate_difference"])
            if pd.isna(r["spd_ci_lower"]) or pd.isna(r["spd_ci_upper"]):
                spd_str = f"{r['statistical_parity_difference']:.3f} (CI n/a)"
            else:
                spd_str = (f"{r['statistical_parity_difference']:.3f} "
                           f"[{r['spd_ci_lower']:.3f}, {r['spd_ci_upper']:.3f}]")
            h_str = f"{r['cohens_h']:.3f} ({r['effect_size_magnitude']})" \
                if not pd.isna(r["cohens_h"]) else "n/a"
            lines.append(
                f"| {r['attribute']} | {r['reference_group']} ({r['reference_group_size']}) | "
                f"{r['compared_group']} ({r['compared_group_size']}) | "
                f"{r['reference_positive_rate']:.3f} | {r['compared_positive_rate']:.3f} | "
                f"{spd_str} | {h_str} | {dir_str} | {eod_str} | {fprd_str} | "
                f"{'YES' if r['four_fifths_rule_flag'] else 'no'} |"
            )
        lines.append("")
        flagged = fairness_df[fairness_df["four_fifths_rule_flag"]]
        if not flagged.empty:
            largest_effect = fairness_df.loc[fairness_df["cohens_h"].abs().idxmax()] \
                if fairness_df["cohens_h"].notna().any() else None
            magnitude_note = ""
            if largest_effect is not None:
                largest_dir_val = largest_effect["disparate_impact_ratio"]
                largest_dir_str = "n/a" if pd.isna(largest_dir_val) else f"{largest_dir_val:.3f}"
                magnitude_note = (
                    f" The largest observed effect size in this run was for "
                    f"`{largest_effect['attribute']}: {largest_effect['compared_group']}` "
                    f"vs. reference `{largest_effect['reference_group']}` "
                    f"(SPD = {largest_effect['statistical_parity_difference']:.3f}, "
                    f"{largest_effect['statistical_parity_difference'] * 100:.1f} percentage "
                    f"points; DIR = {largest_dir_str}; "
                    f"Cohen's h = {largest_effect['cohens_h']:.3f}, "
                    f"{largest_effect['effect_size_magnitude']} effect)."
                )
            lines.append(
                f"**Observed finding:** {len(flagged)} group comparison(s) fall outside the "
                "four-fifths rule bounds, indicating a materially different positive-prediction "
                "rate relative to the reference group for that attribute." + magnitude_note + " "
                "This is an observed statistical pattern in the current data and model; it "
                "does not by itself establish the cause (e.g., it could reflect genuine "
                "differences in the engineered engagement/performance features between "
                "groups, historical outcome disparities encoded in the training labels, or "
                "model bias). Attributing a specific cause would require further "
                "investigation and is not asserted here."
            )
        else:
            lines.append(
                "**Observed finding:** no group comparison in this dataset fell outside the "
                "four-fifths rule bounds. This indicates the model's positive-prediction "
                "rates were relatively close across the observed groups for the evaluated "
                "attributes, under this specific dataset and threshold. This does not "
                "guarantee fairness under other definitions (e.g., equalized odds) or on "
                "other populations."
            )

        eod_available = fairness_df["equal_opportunity_difference"].notna().any()
        fprd_available = fairness_df["false_positive_rate_difference"].notna().any()
        if not eod_available:
            lines.append(
                "\nEqual Opportunity Difference could not be computed for any comparison in "
                "this run — this happens when a group or its reference lacks positive "
                "true-label cases, making TPR undefined for that group."
            )
        if not fprd_available:
            lines.append(
                "\nFalse Positive Rate Difference could not be computed for any comparison in "
                "this run — this happens when a group or its reference lacks negative "
                "true-label cases, making FPR undefined for that group."
            )
    else:
        lines.append(
            "No attribute had two or more groups meeting the minimum sample size threshold "
            f"({MIN_GROUP_SIZE}) with valid positive-prediction rates, so SPD and DIR were "
            "not computed. See `outputs/group_statistics.csv` for the underlying group sizes."
        )
    lines.append("")
    lines.append("![Fairness Comparison](../figures/fairness_comparison.png)")
    lines.append("")

    # ---------------- 6. Limitations ----------------
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append(
        "- **Label bias:** the true label (`psychological_risk`) is a proxy derived from "
        "`final_result` (Withdrawn/Fail), not a clinical psychological assessment. Any "
        "disparity observed reflects disparities in this academic-outcome proxy, not a "
        "validated psychological construct."
    )
    lines.append(
        "- **Reference-group sensitivity:** SPD and DIR values depend on the choice of "
        "reference group (here, the largest observed group per attribute). Different "
        "reference choices can change which comparisons are flagged."
    )
    lines.append(
        "- **Single fairness threshold:** only the default 0.5 probability threshold was "
        "evaluated. Fairness metrics can shift materially at other operating thresholds."
    )
    lines.append(
        "- **No causal claims:** this audit is observational. It reports statistical "
        "association between demographic attributes and model outputs; it does not "
        "establish that any attribute *causes* differences in predictions."
    )
    lines.append(
        "- **Intersectionality not evaluated:** groups are analyzed one attribute at a "
        "time. Intersectional subgroups (e.g., gender × disability) were not evaluated "
        "in this run and may show different patterns than single-attribute groups."
    )
    lines.append(
        "- **Small-sample groups:** groups below the minimum thresholds are reported "
        "with sample counts only, or excluded from ratio metrics, to avoid presenting "
        "statistically unreliable figures; this means some real groups have incomplete "
        "reporting in this audit."
    )
    lines.append(
        "- **Static, single-snapshot audit:** this evaluation reflects the frozen model "
        "and the dataset snapshot at the time of the run. It does not assess drift over "
        "time or across future cohorts."
    )
    lines.append("")

    # ---------------- 7. Ethical considerations ----------------
    lines.append("## 7. Ethical Considerations")
    lines.append("")
    lines.append(
        "- Academic risk predictions of this kind can influence how institutional "
        "resources (advising, outreach, interventions) are allocated to students. "
        "Systematic disparities in positive-prediction rates across demographic groups "
        "could result in unequal access to support, regardless of intent."
    )
    lines.append(
        "- Because the true label is an academic-outcome proxy rather than a clinical "
        "measure, labeling students as psychologically \"at risk\" on this basis alone "
        "carries a risk of mislabeling and should be treated as one input among several "
        "in any human decision process, not an automated determination."
    )
    lines.append(
        "- Disability status and other sensitive attributes evaluated here are protected "
        "characteristics in many jurisdictions' education and anti-discrimination law; "
        "any operational use of this model should involve institutional ethics/legal "
        "review in addition to this technical audit."
    )
    lines.append(
        "- Transparency to affected students about the existence and basis of automated "
        "risk scoring, and a mechanism for human review/appeal, are standard responsible-AI "
        "practices relevant to this type of system."
    )
    lines.append("")

    # ---------------- 8. Recommendations ----------------
    lines.append("## 8. Recommendations")
    lines.append("")
    lines.append(
        "**Based on observed findings in this run:**"
    )
    if not fairness_df.empty and not fairness_df[fairness_df["four_fifths_rule_flag"]].empty:
        lines.append(
            "- Review the specific attribute/group comparisons flagged in Section 5 with "
            "subject-matter and ethics stakeholders before relying on this model's output "
            "for those groups."
        )
    else:
        lines.append(
            "- No four-fifths rule violations were observed in this run; continued periodic "
            "re-auditing is still recommended as the underlying student population changes."
        )
    lines.append(
        "- Report group-wise metrics (Section 4) alongside overall accuracy whenever this "
        "model's outputs are used in decision-making, rather than relying on aggregate "
        "accuracy alone."
    )
    lines.append("")
    lines.append("**Proposed future work (not performed in this audit):**")
    lines.append(
        "- Evaluate additional fairness definitions not covered by this run's SPD, DIR, "
        "EOD, and FPRD metrics — e.g., full equalized odds testing, formal statistical "
        "significance testing (chi-square / Fisher's exact) for each group comparison, "
        "and per-group Expected Calibration Error."
    )
    lines.append(
        "- Evaluate intersectional subgroups where sample sizes permit."
    )
    lines.append(
        "- If disparities are confirmed and judged unacceptable by institutional "
        "stakeholders, consider bias-mitigation approaches (e.g., threshold adjustment "
        "per group, reweighting at retraining time, or feature audit) — any such change "
        "would require retraining and is out of scope for this frozen-model audit."
    )
    lines.append(
        "- Extend this audit to future data snapshots to monitor for fairness drift over "
        "time."
    )
    lines.append("")

    lines.append("---")
    lines.append(
        "*This report was generated entirely from real data and the frozen model's actual "
        "predictions. No group, metric, or conclusion in this report is fabricated or "
        "estimated. Where a metric could not be validly computed, this is stated explicitly "
        "rather than approximated.*"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved report: {report_path}")


# ==================================================================================
# MAIN
# ==================================================================================

def main():
    logger.info("=" * 70)
    logger.info("ETHICS AND FAIRNESS EVALUATION — starting")
    logger.info("=" * 70)

    dirs = ensure_output_dirs()

    model_path = resolve_model_path()
    if model_path is None:
        logger.error(f"Could not locate '{MODEL_FILENAME}' relative to the project structure.")
        logger.error("Set ACADEMIC_RISK_MODEL_PATH to override the path explicitly.")
        sys.exit(1)

    dataset_path = resolve_dataset_path()
    if dataset_path is None:
        logger.error(f"Could not locate '{DATASET_FILENAME}' relative to the project structure.")
        logger.error("Set ACADEMIC_RISK_DATASET_PATH to override the path explicitly.")
        sys.exit(1)

    model = load_model(model_path)
    feature_columns = resolve_model_features(model)
    df = load_dataset(dataset_path)
    validate_required_columns(df, feature_columns)

    demographic_columns, skipped_columns = detect_demographic_columns(df)

    result = run_inference(model, df, feature_columns)

    logger.info("Computing group-wise statistics (with 95% bootstrap CIs where feasible)...")
    group_stats_df = compute_group_statistics(result, demographic_columns)

    logger.info(
        "Computing fairness metrics (SPD, DIR, EOD, FPRD, effect sizes, SPD CIs) "
        "where statistically valid..."
    )
    fairness_df = compute_fairness_metrics(group_stats_df, demographic_columns, result)

    logger.info("Computing calibration by group...")
    calibration_df = compute_calibration_by_group(result, demographic_columns)

    # ---------------- Save tabular outputs ----------------
    group_stats_path = os.path.join(dirs["outputs"], "group_statistics.csv")
    fairness_path = os.path.join(dirs["outputs"], "fairness_metrics.csv")
    calibration_path = os.path.join(dirs["outputs"], "calibration_by_group.csv")

    group_stats_df.to_csv(group_stats_path, index=False)
    logger.info(f"Saved: {group_stats_path}")

    fairness_df.to_csv(fairness_path, index=False)
    logger.info(f"Saved: {fairness_path}")

    calibration_df.to_csv(calibration_path, index=False)
    logger.info(f"Saved: {calibration_path}")

    # ---------------- Figures ----------------
    logger.info("Generating figures (matplotlib, 300 DPI)...")
    fig_demographic_distribution(
        result, demographic_columns, os.path.join(dirs["figures"], "demographic_distribution.png")
    )
    fig_prediction_rate_by_group(
        group_stats_df, os.path.join(dirs["figures"], "prediction_rate_by_group.png")
    )
    fig_fairness_comparison(
        fairness_df, os.path.join(dirs["figures"], "fairness_comparison.png")
    )
    fig_calibration_curves(
        calibration_df, os.path.join(dirs["figures"], "calibration_curves.png")
    )

    # ---------------- Report ----------------
    logger.info("Building fairness report (Markdown)...")
    report_path = os.path.join(dirs["reports"], "fairness_report.md")
    build_report(
        report_path,
        df,
        result,
        demographic_columns,
        skipped_columns,
        group_stats_df,
        fairness_df,
        calibration_df,
        model_path,
        dataset_path,
        feature_columns,
    )

    logger.info("=" * 70)
    logger.info("ETHICS AND FAIRNESS EVALUATION — completed successfully")
    logger.info(f"Outputs directory: {dirs['outputs']}")
    logger.info(f"Figures directory: {dirs['figures']}")
    logger.info(f"Reports directory: {dirs['reports']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
