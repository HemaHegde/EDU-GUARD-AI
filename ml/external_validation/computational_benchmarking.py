"""
EduGuard-AI :: External Validation Pipeline
============================================
Script: computational_benchmarking.py

PURPOSE (strict scope)
-----------------------
Benchmark the COMPUTATIONAL characteristics of the deployed EduGuard-AI
inference pipeline -- using the frozen, already-trained OULAD XGBoost
psychological risk model ONLY -- on both the OULAD training feature
table and the reconstructed ASSISTments 2012-2013 external feature
table.

This is a PERFORMANCE / RESOURCE-USAGE benchmark, not a model
evaluation. It answers "how fast, how heavy, and how consistent is this
model's inference behaviour?", not "is the model's prediction correct?".
Accordingly this script deliberately does NOT compute, import, or
reference:
    - accuracy
    - precision / recall / F1
    - ROC-AUC
    - confusion matrices
    - calibration metrics
    - any other label-dependent evaluation metric
Model behaviour is described only in terms of timing, throughput,
memory, and CPU usage.

Measured quantities:
    - model loading time (repeated, from disk via joblib)
    - prediction latency (per-student, derived from repeated batch runs)
    - throughput (students/sec)
    - SHAP explanation time (TreeExplainer construction + computation)
    - memory usage (resident set size delta per operation)
    - CPU usage (process CPU percent per operation, where available)
    - average latency (and std/min/max) over multiple repeated runs

Run:
    python computational_benchmarking.py

Only the model/input paths below may need editing; everything else is
self-contained. Paths are resolved automatically relative to this
script's location, so the script works regardless of where the
repository is cloned.

Allowed dependencies: pandas, numpy, joblib, shap, matplotlib, pathlib
(standard library `logging`, `time`, `platform`, `sys` for structured
console output and environment reporting). `psutil` is additionally
required for memory and CPU instrumentation (`pip install psutil`).
No seaborn. No retraining. No evaluation metrics.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: safe for headless / CI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("computational_benchmarking")


# --------------------------------------------------------------------------
# PROJECT ROOT RESOLUTION (works regardless of clone location / cwd)
# --------------------------------------------------------------------------
# This script lives at: <project_root>/ml/external_validation/computational_benchmarking.py
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

# OULAD training population behavioural feature table.
OULAD_DATA_PATH = PROJECT_ROOT / "ml" / "final_student_psychology_dataset.csv"

# Reconstructed ASSISTments external behavioural feature table, produced
# by preprocess_assistments.py. Benchmarking uses the raw feature table
# (not the pre-computed predictions file), since predictions are
# generated fresh here as part of the timed workload.
ASSISTMENTS_DATA_PATH = (
    PROJECT_ROOT / "datasets" / "ASSISTments" / "assistments_student_profiles.csv"
)

# Output locations.
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
REPORT_DIR = SCRIPT_DIR / "reports"

RAW_RUNS_PATH = OUTPUT_DIR / "benchmark_raw_runs.csv"
SUMMARY_PATH = OUTPUT_DIR / "benchmark_summary.csv"
SYSTEM_INFO_PATH = OUTPUT_DIR / "system_info.csv"

LATENCY_FIG_PATH = FIGURE_DIR / "latency_overview.png"
THROUGHPUT_FIG_PATH = FIGURE_DIR / "throughput_comparison.png"
RESOURCE_FIG_PATH = FIGURE_DIR / "memory_cpu_usage.png"

REPORT_PATH = REPORT_DIR / "computational_benchmarking_report.md"

# --------------------------------------------------------------------------
# FIXED FEATURE SCHEMA
# --------------------------------------------------------------------------
# These seven columns are FIXED by the original OULAD training pipeline
# (see preprocess_oulad.py) and are the exact feature set the frozen
# model expects.
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
# BENCHMARK CONFIGURATION
# --------------------------------------------------------------------------
# Number of repeated, timed runs per operation. Repeating each operation
# and reporting mean/median/std/min/max (rather than a single timing) is
# what makes these numbers robust to transient OS scheduling noise.
N_MODEL_LOAD_RUNS = 10
N_PREDICTION_RUNS = 20
N_SHAP_RUNS = 5

# One untimed warm-up call is performed before each timed loop, to avoid
# attributing one-off cold-start costs (e.g. first-call JIT / cache
# effects inside XGBoost) to the steady-state latency figures. Warm-up
# calls are explicitly excluded from all reported statistics.
N_WARMUP_RUNS = 1

# Row caps keep the benchmark tractable and comparable across
# populations of very different sizes. Rows are subsampled with a fixed
# random_state for reproducibility; SHAP benchmarking uses a smaller cap
# than prediction benchmarking because TreeSHAP is far more expensive
# per row.
MAX_ROWS_FOR_PREDICTION_BENCHMARK = 20_000
MAX_ROWS_FOR_SHAP_BENCHMARK = 300
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# LOW-LEVEL MEASUREMENT HELPER
# --------------------------------------------------------------------------

def _measure(func: Callable[[], object]) -> tuple[object, float, float, float]:
    """
    Execute `func` once and measure wall-clock elapsed time, process CPU
    percent, and resident-memory delta attributable to that single call.

    CPU percent is measured using psutil's non-blocking convention: an
    un-timed priming call to `cpu_percent(interval=None)` establishes a
    baseline, and the value returned immediately after `func` executes
    reflects CPU utilisation accumulated since that baseline.

    Returns (result, elapsed_seconds, cpu_percent, memory_delta_mb).
    """
    process = psutil.Process()
    process.cpu_percent(interval=None)  # prime the internal baseline
    mem_before = process.memory_info().rss

    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0

    cpu_percent = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss
    memory_delta_mb = (mem_after - mem_before) / (1024 ** 2)

    return result, elapsed, cpu_percent, memory_delta_mb


# --------------------------------------------------------------------------
# STEP HELPERS
# --------------------------------------------------------------------------

def load_dataset(path: Path, name: str) -> pd.DataFrame:
    """Load a behavioural feature table from disk. Fails fast if absent."""
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found at: {path}")
    logger.info(f"Loading {name} dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} {name} rows.")
    return df


def validate_feature_columns(df: pd.DataFrame, name: str) -> None:
    """Verify all seven required behavioural feature columns are present. Fails fast."""
    missing_columns = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The {name} dataset is missing required feature column(s): "
            f"{missing_columns}. Expected all of: {REQUIRED_FEATURE_COLUMNS}."
        )
    logger.info(f"All seven required feature columns verified present in {name} dataset.")


def prepare_feature_matrix(df: pd.DataFrame, name: str, max_rows: int) -> pd.DataFrame:
    """
    Extract, order, and (if needed) subsample the fixed seven-feature
    matrix in the exact column order the frozen model was trained on.
    """
    X = df[REQUIRED_FEATURE_COLUMNS].copy()

    if len(X) > max_rows:
        logger.info(
            f"{name} dataset has {len(X)} rows; subsampling to {max_rows} rows "
            f"(random_state={RANDOM_STATE}) for tractable, comparable benchmarking."
        )
        X = X.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)

    return X


def benchmark_model_loading(path: Path, n_runs: int):
    """
    Repeatedly load the frozen model from disk via joblib, timing each
    load. Returns the last-loaded model instance (used for all
    downstream inference benchmarks) plus a list of per-run records.

    The model is loaded in inference mode only on every run -- `.fit()`
    is never called anywhere in this script.
    """
    logger.info(f"Benchmarking model loading ({n_runs} runs) from: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at: {path}\n"
            "Ensure academic_risk_xgboost.pkl has been generated and "
            "placed at the expected location."
        )

    records = []
    model = None
    for i in range(n_runs):
        result, elapsed, cpu_pct, mem_delta = _measure(lambda: joblib.load(path))
        model = result
        records.append({
            "operation": "model_loading",
            "dataset": "not_applicable",
            "run_index": i,
            "elapsed_seconds": elapsed,
            "cpu_percent": cpu_pct,
            "memory_delta_mb": mem_delta,
            "n_rows": np.nan,
            "throughput_students_per_sec": np.nan,
        })
        logger.info(f"  [load run {i + 1}/{n_runs}] elapsed={elapsed * 1000:.2f} ms")

    logger.info(f"Model loading benchmark complete. Model type: {type(model).__name__}")
    return model, records


def benchmark_prediction(model, X: pd.DataFrame, name: str, n_runs: int, n_warmup: int):
    """
    Benchmark repeated batch inference (predict_proba) on a fixed
    feature matrix. Each run scores the full matrix in one batch; per-
    run throughput (students/sec) and per-student latency (ms) are
    derived from the batch elapsed time.

    Inference-only: `model.predict_proba` never mutates or refits the
    model.
    """
    logger.info(f"Benchmarking prediction latency for {name} (n_rows={len(X)}, {n_runs} runs) ...")

    for _ in range(n_warmup):
        model.predict_proba(X)  # untimed warm-up, excluded from statistics

    records = []
    n_rows = len(X)
    for i in range(n_runs):
        _, elapsed, cpu_pct, mem_delta = _measure(lambda: model.predict_proba(X))
        throughput = n_rows / elapsed if elapsed > 0 else float("nan")
        records.append({
            "operation": "prediction",
            "dataset": name,
            "run_index": i,
            "elapsed_seconds": elapsed,
            "cpu_percent": cpu_pct,
            "memory_delta_mb": mem_delta,
            "n_rows": n_rows,
            "throughput_students_per_sec": throughput,
        })
        logger.info(
            f"  [{name} predict run {i + 1}/{n_runs}] elapsed={elapsed * 1000:.2f} ms, "
            f"throughput={throughput:.1f} students/sec"
        )

    logger.info(f"Prediction latency benchmark complete for {name}.")
    return records


def benchmark_shap_explainer_construction(model, X: pd.DataFrame, name: str, n_runs: int, n_warmup: int):
    """
    Benchmark repeated construction of a `shap.TreeExplainer` wrapping
    the frozen model. Explainer construction cost is independent of the
    feature matrix itself, but is measured once per population for
    completeness / consistency of reporting.
    """
    logger.info(f"Benchmarking SHAP explainer construction for {name} ({n_runs} runs) ...")

    for _ in range(n_warmup):
        shap.TreeExplainer(model)  # untimed warm-up

    records = []
    for i in range(n_runs):
        _, elapsed, cpu_pct, mem_delta = _measure(lambda: shap.TreeExplainer(model))
        records.append({
            "operation": "shap_explainer_construction",
            "dataset": name,
            "run_index": i,
            "elapsed_seconds": elapsed,
            "cpu_percent": cpu_pct,
            "memory_delta_mb": mem_delta,
            "n_rows": np.nan,
            "throughput_students_per_sec": np.nan,
        })
        logger.info(f"  [{name} explainer-build run {i + 1}/{n_runs}] elapsed={elapsed * 1000:.2f} ms")

    return records


def benchmark_shap_computation(explainer, X: pd.DataFrame, name: str, n_runs: int, n_warmup: int):
    """
    Benchmark repeated TreeSHAP value computation on a fixed, capped
    feature matrix using a single pre-built explainer instance.
    """
    logger.info(f"Benchmarking SHAP value computation for {name} (n_rows={len(X)}, {n_runs} runs) ...")

    for _ in range(n_warmup):
        explainer(X)  # untimed warm-up

    records = []
    n_rows = len(X)
    for i in range(n_runs):
        _, elapsed, cpu_pct, mem_delta = _measure(lambda: explainer(X))
        throughput = n_rows / elapsed if elapsed > 0 else float("nan")
        records.append({
            "operation": "shap_value_computation",
            "dataset": name,
            "run_index": i,
            "elapsed_seconds": elapsed,
            "cpu_percent": cpu_pct,
            "memory_delta_mb": mem_delta,
            "n_rows": n_rows,
            "throughput_students_per_sec": throughput,
        })
        logger.info(
            f"  [{name} SHAP-compute run {i + 1}/{n_runs}] elapsed={elapsed * 1000:.2f} ms, "
            f"throughput={throughput:.1f} students/sec"
        )

    logger.info(f"SHAP value computation benchmark complete for {name}.")
    return records


def summarize_runs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run measurements into mean/median/std/min/max summary
    statistics, grouped by (operation, dataset). This is the primary
    table for reporting "average latency over multiple runs".
    """
    grouped = raw_df.groupby(["operation", "dataset"], sort=False)

    summary = grouped.agg(
        n_runs=("run_index", "count"),
        mean_elapsed_seconds=("elapsed_seconds", "mean"),
        median_elapsed_seconds=("elapsed_seconds", "median"),
        std_elapsed_seconds=("elapsed_seconds", "std"),
        min_elapsed_seconds=("elapsed_seconds", "min"),
        max_elapsed_seconds=("elapsed_seconds", "max"),
        mean_cpu_percent=("cpu_percent", "mean"),
        mean_memory_delta_mb=("memory_delta_mb", "mean"),
        mean_throughput_students_per_sec=("throughput_students_per_sec", "mean"),
        mean_n_rows=("n_rows", "mean"),
    ).reset_index()

    summary["std_elapsed_seconds"] = summary["std_elapsed_seconds"].fillna(0.0)
    summary["mean_latency_ms_per_student"] = np.where(
        summary["mean_n_rows"] > 0,
        (summary["mean_elapsed_seconds"] / summary["mean_n_rows"]) * 1000.0,
        np.nan,
    )

    logger.info("Aggregated run-level statistics computed for all benchmarked operations.")
    return summary


def collect_system_info() -> pd.DataFrame:
    """
    Collect environment / hardware metadata relevant to reproducing the
    benchmark: platform, CPU, memory, and key library versions.
    """
    import xgboost

    vm = psutil.virtual_memory()
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "total_memory_gb": round(vm.total / (1024 ** 3), 2),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "shap_version": shap.__version__,
        "xgboost_version": xgboost.__version__,
        "psutil_version": psutil.__version__,
    }
    logger.info(
        f"System info: {info['platform']}, "
        f"{info['cpu_logical_cores']} logical cores, "
        f"{info['total_memory_gb']} GB RAM."
    )
    return pd.DataFrame([info])


# --------------------------------------------------------------------------
# FIGURE HELPERS
# --------------------------------------------------------------------------

def plot_latency_overview(summary: pd.DataFrame, path: Path) -> None:
    """Publication-quality bar chart of mean elapsed time (+/- std) per operation, log-scaled."""
    plot_df = summary.copy()
    plot_df["label"] = plot_df["operation"] + "\n(" + plot_df["dataset"] + ")"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(plot_df))
    ax.bar(
        x, plot_df["mean_elapsed_seconds"], yerr=plot_df["std_elapsed_seconds"],
        color="#4C72B0", edgecolor="white", linewidth=0.5, capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], fontsize=8, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Elapsed Time (seconds, log scale)", fontsize=10)
    ax.set_title(
        "Computational Benchmark — Mean Elapsed Time per Operation\n"
        "(error bars = standard deviation across repeated runs)",
        fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="both", color="lightgrey", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_throughput_comparison(summary: pd.DataFrame, path: Path) -> None:
    """Publication-quality bar chart of mean throughput for row-scaled operations."""
    plot_df = summary[summary["mean_throughput_students_per_sec"].notna()].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = plot_df["operation"] + " — " + plot_df["dataset"]
    bars = ax.bar(
        labels, plot_df["mean_throughput_students_per_sec"],
        color="#55A868", edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, plot_df["mean_throughput_students_per_sec"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Throughput (students/sec)", fontsize=10)
    ax.set_title("Computational Benchmark — Mean Throughput by Operation", fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_resource_usage(summary: pd.DataFrame, path: Path) -> None:
    """Publication-quality dual-panel chart of mean memory delta and CPU percent per operation."""
    plot_df = summary.copy()
    plot_df["label"] = plot_df["operation"] + "\n(" + plot_df["dataset"] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(plot_df["label"], plot_df["mean_memory_delta_mb"],
                color="#C44E52", edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("Mean Memory Delta (MB)", fontsize=10)
    axes[0].set_title("Memory Usage per Operation", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=8, labelrotation=25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(plot_df["label"], plot_df["mean_cpu_percent"],
                color="#8172B2", edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Mean CPU Usage (%)", fontsize=10)
    axes[1].set_title("CPU Usage per Operation", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=8, labelrotation=25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Computational Benchmark — Resource Usage", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------------------------

def generate_report(
    summary: pd.DataFrame,
    system_info: pd.DataFrame,
    n_oulad_rows: int,
    n_assistments_rows: int,
    path: Path,
) -> None:
    """Write a markdown report documenting the benchmarking methodology and results."""

    sys_row = system_info.iloc[0]
    sys_table = (
        "| Property | Value |\n|---|---|\n"
        f"| Platform | {sys_row['platform']} |\n"
        f"| Python version | {sys_row['python_version']} |\n"
        f"| Logical CPU cores | {sys_row['cpu_logical_cores']} |\n"
        f"| Physical CPU cores | {sys_row['cpu_physical_cores']} |\n"
        f"| Total system memory | {sys_row['total_memory_gb']} GB |\n"
        f"| numpy | {sys_row['numpy_version']} |\n"
        f"| pandas | {sys_row['pandas_version']} |\n"
        f"| joblib | {sys_row['joblib_version']} |\n"
        f"| shap | {sys_row['shap_version']} |\n"
        f"| xgboost | {sys_row['xgboost_version']} |\n"
        f"| psutil | {sys_row['psutil_version']} |"
    )

    summary_table_lines = [
        "| Operation | Dataset | n Runs | Mean (s) | Median (s) | Std (s) | Min (s) | Max (s) | "
        "Mean Latency (ms/student) | Mean Throughput (students/sec) | Mean CPU (%) | Mean Memory Δ (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        latency_str = f"{row['mean_latency_ms_per_student']:.4f}" if pd.notna(row["mean_latency_ms_per_student"]) else "n/a"
        throughput_str = f"{row['mean_throughput_students_per_sec']:,.1f}" if pd.notna(row["mean_throughput_students_per_sec"]) else "n/a"
        summary_table_lines.append(
            f"| {row['operation']} | {row['dataset']} | {int(row['n_runs'])} | "
            f"{row['mean_elapsed_seconds']:.6f} | {row['median_elapsed_seconds']:.6f} | "
            f"{row['std_elapsed_seconds']:.6f} | {row['min_elapsed_seconds']:.6f} | "
            f"{row['max_elapsed_seconds']:.6f} | {latency_str} | {throughput_str} | "
            f"{row['mean_cpu_percent']:.2f} | {row['mean_memory_delta_mb']:.3f} |"
        )
    summary_table = "\n".join(summary_table_lines)

    report = f"""# Computational Benchmarking Report

## Purpose

This report documents the computational performance of the **frozen,
OULAD-trained XGBoost psychological risk model** (`academic_risk_xgboost.pkl`)
across model loading, batch prediction, and TreeSHAP explanation, on
both its OULAD training population and the external ASSISTments
population. It is a **resource-usage and latency benchmark**, not a
model evaluation: no accuracy, precision, recall, F1, ROC-AUC,
confusion-matrix, or calibration quantity is computed anywhere in this
script, and none should be inferred from it.

## Methodology

- The frozen model was loaded via `joblib.load()` in inference mode
  only, {N_MODEL_LOAD_RUNS} times in a row, to characterise load-time
  variability. It was never retrained, fine-tuned, or refit; `.fit()`
  is never called anywhere in this script.
- Prediction benchmarking used `model.predict_proba()` on the fixed
  seven-feature matrix for each population ({N_PREDICTION_RUNS} timed
  batch runs each, after {N_WARMUP_RUNS} untimed warm-up run(s) to
  exclude one-off cold-start cost from the reported statistics).
- SHAP benchmarking used a single shared `shap.TreeExplainer` wrapping
  the frozen model (construction timed separately, {N_SHAP_RUNS} runs
  each), with value computation timed on a capped, fixed subsample of
  up to {MAX_ROWS_FOR_SHAP_BENCHMARK} rows per population.
- Every timed operation is preceded by an untimed warm-up call; warm-up
  calls are excluded from all reported statistics.
- Memory usage is the process resident-set-size (RSS) delta attributable
  to a single call, in megabytes. CPU usage is the process CPU percent
  accumulated since the previous measurement point, as reported by
  `psutil`.
- All statistics are aggregated as mean, median, standard deviation,
  minimum, and maximum across the repeated runs of each operation.

## Populations Benchmarked

| Population | Rows available | Rows used for prediction benchmark | Rows used for SHAP benchmark |
|---|---|---|---|
| OULAD (training) | {n_oulad_rows} | {min(n_oulad_rows, MAX_ROWS_FOR_PREDICTION_BENCHMARK)} | {min(n_oulad_rows, MAX_ROWS_FOR_SHAP_BENCHMARK)} |
| ASSISTments (external) | {n_assistments_rows} | {min(n_assistments_rows, MAX_ROWS_FOR_PREDICTION_BENCHMARK)} | {min(n_assistments_rows, MAX_ROWS_FOR_SHAP_BENCHMARK)} |

## System / Environment Information

{sys_table}

## Benchmark Results

{summary_table}

**Reading this table:** "Mean Latency (ms/student)" and "Mean
Throughput (students/sec)" are only meaningful for row-scaled operations
(prediction, SHAP value computation) and are reported as `n/a` for
model loading and SHAP explainer construction, which do not scale with
row count.

## Figures

- `figures/latency_overview.png` -- mean elapsed time (± std) per
  operation, log-scaled
- `figures/throughput_comparison.png` -- mean throughput (students/sec)
  for row-scaled operations
- `figures/memory_cpu_usage.png` -- mean memory delta and mean CPU
  usage per operation

## Scope Note

This report and its underlying script intentionally exclude any
assessment of predictive correctness. All figures describe the
computational cost of running the frozen model's inference and
explanation pipeline -- they say nothing about whether the model's
outputs are accurate for either population, which is outside the scope
of this pipeline stage and cannot be assessed here without ground-truth
labels.
"""

    path.write_text(report, encoding="utf-8")
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("EduGuard-AI :: Computational Benchmarking (Frozen Model, Inference Only)")
        logger.info("=" * 70)

        all_records: list[dict] = []

        # ---- 1. Model loading benchmark ----
        model, load_records = benchmark_model_loading(MODEL_PATH, N_MODEL_LOAD_RUNS)
        all_records.extend(load_records)

        # ---- 2. Load feature tables and validate schema (fail-fast) ----
        oulad_df = load_dataset(OULAD_DATA_PATH, "OULAD")
        assistments_df = load_dataset(ASSISTMENTS_DATA_PATH, "ASSISTments")
        validate_feature_columns(oulad_df, "OULAD")
        validate_feature_columns(assistments_df, "ASSISTments")

        # ---- 3. Prediction latency + throughput benchmarks ----
        X_oulad_pred = prepare_feature_matrix(oulad_df, "OULAD", MAX_ROWS_FOR_PREDICTION_BENCHMARK)
        X_assistments_pred = prepare_feature_matrix(assistments_df, "ASSISTments", MAX_ROWS_FOR_PREDICTION_BENCHMARK)

        all_records.extend(
            benchmark_prediction(model, X_oulad_pred, "OULAD", N_PREDICTION_RUNS, N_WARMUP_RUNS)
        )
        all_records.extend(
            benchmark_prediction(model, X_assistments_pred, "ASSISTments", N_PREDICTION_RUNS, N_WARMUP_RUNS)
        )

        # ---- 4. SHAP explanation time benchmarks ----
        X_oulad_shap = prepare_feature_matrix(oulad_df, "OULAD", MAX_ROWS_FOR_SHAP_BENCHMARK)
        X_assistments_shap = prepare_feature_matrix(assistments_df, "ASSISTments", MAX_ROWS_FOR_SHAP_BENCHMARK)

        all_records.extend(
            benchmark_shap_explainer_construction(model, X_oulad_shap, "OULAD", N_SHAP_RUNS, N_WARMUP_RUNS)
        )
        all_records.extend(
            benchmark_shap_explainer_construction(model, X_assistments_shap, "ASSISTments", N_SHAP_RUNS, N_WARMUP_RUNS)
        )

        # A single shared explainer per population is used for the value
        # -computation timing loop, consistent with normal deployment
        # usage (build once, explain many).
        oulad_explainer = shap.TreeExplainer(model)
        all_records.extend(
            benchmark_shap_computation(oulad_explainer, X_oulad_shap, "OULAD", N_SHAP_RUNS, N_WARMUP_RUNS)
        )
        assistments_explainer = shap.TreeExplainer(model)
        all_records.extend(
            benchmark_shap_computation(assistments_explainer, X_assistments_shap, "ASSISTments", N_SHAP_RUNS, N_WARMUP_RUNS)
        )

        # ---- 5. Persist raw per-run measurements ----
        raw_df = pd.DataFrame(all_records)
        raw_df.to_csv(RAW_RUNS_PATH, index=False)
        logger.info(f"[SAVED] {RAW_RUNS_PATH}")

        # ---- 6. Aggregate summary statistics ----
        summary_df = summarize_runs(raw_df)
        summary_df.to_csv(SUMMARY_PATH, index=False)
        logger.info(f"[SAVED] {SUMMARY_PATH}")

        # ---- 7. System / environment info ----
        system_info_df = collect_system_info()
        system_info_df.to_csv(SYSTEM_INFO_PATH, index=False)
        logger.info(f"[SAVED] {SYSTEM_INFO_PATH}")

        # ---- 8. Figures ----
        logger.info("Generating publication-quality figures ...")
        plot_latency_overview(summary_df, LATENCY_FIG_PATH)
        plot_throughput_comparison(summary_df, THROUGHPUT_FIG_PATH)
        plot_resource_usage(summary_df, RESOURCE_FIG_PATH)

        # ---- 9. Markdown report ----
        generate_report(
            summary_df, system_info_df,
            n_oulad_rows=len(oulad_df), n_assistments_rows=len(assistments_df),
            path=REPORT_PATH,
        )

        logger.info("=" * 70)
        logger.info("Computational benchmarking complete.")
        logger.info("=" * 70)

    except (FileNotFoundError, ValueError) as exc:
        # Fail loudly and clearly rather than proceeding with invalid or
        # incomplete data / model state.
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
