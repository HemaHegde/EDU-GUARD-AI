"""
EduGuard-AI :: External Validation Pipeline
============================================
Preprocessing script: ASSISTments 2012-2013 -> per-student behavioural
profiles mapped onto the seven FIXED OULAD training features used by the
EduGuard-AI XGBoost psychological risk model.

SCOPE (strict):
    This script implements ONLY data loading, cleaning, and feature
    engineering. It deliberately does NOT:
        - train any model
        - run any prediction
        - compute SHAP values
        - perform any evaluation / metrics

Run:
    python preprocess_assistments.py

Only the dataset path below may need editing; everything else is
self-contained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/external_validation/preprocess_assistments.py
# so the project root is two directories above this file. Resolving it
# from __file__ (rather than the current working directory) means the
# script behaves identically whether it's run as
# `python preprocess_assistments.py` from inside its own folder, or
# invoked from anywhere else (e.g. `python ml/external_validation/preprocess_assistments.py`).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # ml/external_validation -> ml -> project_root

# Path to the raw ASSISTments 2012-2013 export.
INPUT_PATH = PROJECT_ROOT / "datasets" / "ASSISTments" / "2012-2013-data-with-predictions-4-final.csv"

# Required output location (per spec).
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "ASSISTments" / "assistments_student_profiles.csv"

# --- Documented rule for "clearly impossible" duration outliers ----------
# A single problem-log interaction lasting longer than this is treated as
# a logging/session artefact (e.g. idle tab, browser left open overnight)
# rather than genuine continuous engagement, and is discarded. 3 hours is
# used as a conservative upper bound for a single problem-log interaction,
# consistent with typical single-session lengths reported in ASSISTments
# learning-analytics literature.
MAX_DURATION_SECONDS = 3 * 60 * 60  # 3 hours = 10,800 seconds

# --- Documented rule for resolving the "class" grouping key --------------
# The spec requires grouping by (student_class_id, user_id) BEFORE
# computing temporal features, so that inactivity is never measured across
# unrelated class enrolments. The raw ASSISTments 2012-2013 export does
# not always ship a column literally named `student_class_id`. The
# following candidates are checked, in priority order, before falling
# back to a single synthetic class id (see _resolve_class_column).
CLASS_COLUMN_CANDIDATES = ["student_class_id", "class_id", "context_id", "sequence_id"]

# Columns that must exist in the raw file for this pipeline to run.
REQUIRED_COLUMNS = ["user_id", "start_time", "end_time", "correct", "attempt_count", "hint_count"]


# --------------------------------------------------------------------------
# STEP HELPERS
# --------------------------------------------------------------------------

def _resolve_class_column(df: pd.DataFrame) -> str:
    """
    Resolve which column should be used as the class-grouping key.

    Documented assumption:
    If no plausible class-identifying column is found, we fall back to a
    single synthetic class label ('UNKNOWN_CLASS') applied to every row.
    This means temporal features would then be computed per-student only
    (no cross-class separation is possible without real class metadata).
    This fallback is logged explicitly -- it is never a silent default.
    """
    for col in CLASS_COLUMN_CANDIDATES:
        if col in df.columns:
            print(f"[INFO] Using '{col}' as the class-grouping key.")
            return col

    print(
        "[WARNING] No class-identifying column found among candidates "
        f"{CLASS_COLUMN_CANDIDATES}. Falling back to a single synthetic "
        "class 'UNKNOWN_CLASS' per student. This is a documented "
        "approximation made explicit in the logs, not a silent default."
    )
    df["student_class_id"] = "UNKNOWN_CLASS"
    return "student_class_id"


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw ASSISTments CSV."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Raw rows loaded: {len(df)}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Fail fast if required raw columns are absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def parse_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Parse start_time / end_time into datetime objects.

    Rows where either timestamp fails to parse (becomes NaT) are removed
    and counted as invalid timestamps.
    """
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    invalid_mask = df["start_time"].isna() | df["end_time"].isna()
    n_invalid = int(invalid_mask.sum())

    df = df.loc[~invalid_mask].copy()
    return df, n_invalid


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows (full-row duplicates)."""
    before = len(df)
    df = df.drop_duplicates()
    n_removed = before - len(df)
    return df, n_removed


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction duration in seconds (end_time - start_time)."""
    df = df.copy()
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    return df


def remove_negative_durations(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows where end_time precedes start_time (impossible)."""
    before = len(df)
    df = df.loc[df["duration"] >= 0].copy()
    n_removed = before - len(df)
    return df, n_removed


def remove_duration_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove clearly impossible durations using the documented rule:
    any single interaction longer than MAX_DURATION_SECONDS is dropped
    as a logging/session artefact (see CONFIGURATION section above).
    """
    before = len(df)
    df = df.loc[df["duration"] <= MAX_DURATION_SECONDS].copy()
    n_removed = before - len(df)
    return df, n_removed


def ensure_numeric_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce feature-relevant raw columns to numeric and apply row-level
    imputation BEFORE aggregation.

    Documented row-level imputation policy:
    - attempt_count: non-numeric / missing -> 0
        (treated as "no recorded attempt activity" for that log row,
        consistent with the zero-imputation convention used for the
        original OULAD behavioural features).
    - correct: non-numeric / missing -> left as NaN.
        We deliberately do NOT fabricate a correctness value (e.g. 0 or
        1) for missing entries; pandas' NaN-skipping mean()/std() are
        used downstream so these rows simply do not influence
        avg_score / assessment_consistency.
    """
    df = df.copy()
    df["attempt_count"] = pd.to_numeric(df["attempt_count"], errors="coerce").fillna(0)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# FEATURE ENGINEERING (per student, within a class group)
# --------------------------------------------------------------------------

def compute_student_profile(group: pd.DataFrame) -> pd.Series:
    """
    Compute the seven fixed OULAD-mapped behavioural features for a single
    (student_class_id, user_id) group. The observation window is scoped
    strictly to this group, so inactivity/engagement-slope calculations
    never leak across unrelated class enrolments.
    """
    # --- 1. total_clicks: SUM(attempt_count); hint_count is excluded ----
    total_clicks = group["attempt_count"].sum()

    # --- 2. avg_score: MEAN(correct) x 100, NaNs skipped automatically --
    # NOTE ON SCALE: OULAD's avg_score is a mean assessment score on a
    # 0-100 scale (see preprocess_oulad.py: mean of studentAssessment
    # `score`, which OULAD reports as a percentage). ASSISTments'
    # `correct` field is a 0/1 binary correctness indicator, so its raw
    # mean is a 0-1 proportion. Multiplying by 100 here converts that
    # proportion to a percentage, aligning both datasets on the same
    # 0-100 scale. Without this conversion, avg_score values from the
    # two datasets are not comparable (e.g. OULAD 72.5 vs ASSISTments
    # 0.725 for equivalent performance), which would artificially
    # inflate any downstream distributional-shift metric (KS statistic,
    # PSI) computed on this feature.
    avg_score = group["correct"].mean() * 100

    # Daily aggregation used by features 3-6 below.
    group = group.copy()
    group["date"] = group["start_time"].dt.normalize()
    daily_attempts = group.groupby("date")["attempt_count"].sum().sort_index()

    # --- 3. active_days: COUNT(DISTINCT DATE(start_time)) ---------------
    active_days = int(daily_attempts.shape[0])

    # --- 4. engagement_variability: STD(daily_attempts) ------------------
    # Population std (ddof=0) is used so a student with only one active
    # day yields 0.0 rather than NaN (sample std is undefined for n=1).
    engagement_variability = float(daily_attempts.std(ddof=0)) if active_days > 0 else 0.0

    # --- 5. inactivity_days: (last_day - first_day) - active_days, >=0 --
    if active_days > 0:
        span_days = (daily_attempts.index.max() - daily_attempts.index.min()).days
        inactivity_days = max(span_days - active_days, 0)
    else:
        inactivity_days = 0

    # --- 6. engagement_slope: OLS vs. actual elapsed calendar day -------
    if active_days >= 2:
        elapsed_days = (daily_attempts.index - daily_attempts.index.min()).days.values.astype(float)
        clicks = daily_attempts.values.astype(float)
        x_mean = elapsed_days.mean()
        y_mean = clicks.mean()
        numerator = np.sum((elapsed_days - x_mean) * (clicks - y_mean))
        denominator = np.sum((elapsed_days - x_mean) ** 2)
        engagement_slope = float(numerator / denominator) if denominator != 0 else 0.0
    else:
        # A single active day carries no temporal trend information.
        engagement_slope = 0.0

    # --- 7. assessment_consistency: STD(correct) -------------------------
    assessment_consistency = group["correct"].std(ddof=0)

    return pd.Series(
        {
            "total_clicks": total_clicks,
            "avg_score": avg_score,
            "active_days": active_days,
            "engagement_variability": engagement_variability,
            "inactivity_days": inactivity_days,
            "engagement_slope": engagement_slope,
            "assessment_consistency": assessment_consistency,
        }
    )


def build_student_profiles(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    """
    Group by (class_col, user_id) and compute the seven behavioural
    features per group. Grouping by class first (per spec) prevents
    inactivity_days / engagement_slope from being inflated by gaps
    between unrelated class enrolments for the same student.
    """
    profiles = (
        df.groupby([class_col, "user_id"], sort=False)
        .apply(compute_student_profile)
        .reset_index()
    )
    profiles = profiles.rename(columns={class_col: "student_class_id", "user_id": "student_id"})
    return profiles


def impute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final missing-value pass on the aggregated student-level feature
    table.

    Documented imputation policy (applied after aggregation):
    - avg_score: missing (student had no rows with a valid `correct`
      value) -> 0.0, treated as zero observed performance. This mirrors
      the zero-imputation convention used for the original OULAD
      behavioural features in the source framework.
    - assessment_consistency: missing (population std is mathematically
      0.0 only when defined; true NaN here means no graded rows existed
      at all) -> 0.0, i.e. no observed variability.
    - engagement_variability: missing -> 0.0, no observed variability.
    - total_clicks, active_days, inactivity_days, engagement_slope:
      should not be NaN given the aggregation logic above, but are
      defensively filled with 0.0 for robustness against edge cases.
    """
    df = df.copy()
    fill_values = {
        "total_clicks": 0.0,
        "avg_score": 0.0,
        "active_days": 0.0,
        "engagement_variability": 0.0,
        "inactivity_days": 0.0,
        "engagement_slope": 0.0,
        "assessment_consistency": 0.0,
    }
    n_missing_before = int(df[list(fill_values.keys())].isna().sum().sum())
    df = df.fillna(value=fill_values)
    print(f"[INFO] Total missing feature values imputed (post-aggregation): {n_missing_before}")
    return df


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    # ---- Load ----
    df = load_data(INPUT_PATH)
    n_rows_raw = len(df)
    validate_columns(df)

    # ---- 2-3. Parse timestamps, drop invalid ----
    df, n_invalid_timestamps = parse_timestamps(df)

    # ---- 4. Remove duplicate rows ----
    df, n_duplicates_removed = remove_duplicates(df)

    # ---- 5. Compute duration ----
    df = compute_duration(df)

    # ---- 6. Remove negative durations ----
    df, n_negative_durations = remove_negative_durations(df)

    # ---- 7. Remove clearly impossible duration outliers ----
    df, n_duration_outliers = remove_duration_outliers(df)

    # ---- Resolve class-grouping column (documented fallback logic) ----
    class_col = _resolve_class_column(df)

    # ---- Row-level numeric coercion + imputation ----
    df = ensure_numeric_inputs(df)

    # ---- Build per-student profiles (grouped by class, then student) ----
    profiles = build_student_profiles(df, class_col)

    # ---- Final feature-level imputation ----
    profiles = impute_missing_features(profiles)

    # ---- Enforce final column order ----
    final_columns = [
        "student_id",
        "student_class_id",
        "total_clicks",
        "avg_score",
        "active_days",
        "engagement_variability",
        "inactivity_days",
        "engagement_slope",
        "assessment_consistency",
    ]
    profiles = profiles[final_columns]

    # ---- Save output ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(OUTPUT_PATH, index=False)

    # ---- Logging summary (per spec) ----
    n_removed_rows = (
        n_invalid_timestamps
        + n_duplicates_removed
        + n_negative_durations
        + n_duration_outliers
    )
    n_students = profiles["student_id"].nunique()
    n_final_profiles = len(profiles)

    print("\n" + "=" * 60)
    print("EduGuard-AI :: ASSISTments preprocessing summary")
    print("=" * 60)
    print(f"Raw rows loaded:                  {n_rows_raw}")
    print(f"Number of removed rows (total):   {n_removed_rows}")
    print(f"  - invalid timestamps removed:   {n_invalid_timestamps}")
    print(f"  - duplicate rows removed:       {n_duplicates_removed}")
    print(f"  - negative-duration rows:       {n_negative_durations}")
    print(f"  - duration outlier rows:        {n_duration_outliers}")
    print(f"Number of unique students:        {n_students}")
    print(f"Number of final student profiles: {n_final_profiles}")
    print(f"Output saved to:                  {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
