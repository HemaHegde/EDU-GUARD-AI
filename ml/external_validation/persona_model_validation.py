"""
EduGuard AI — External Validation of the Deployed Persona Engine
==================================================================
Validates the FROZEN, already-deployed persona clustering artifacts
(persona_scaler.pkl + persona_kmeans.pkl) exactly as they are used in
production by advanced_persona_engine.py.

This script does NOT train, fit, or re-fit anything. It only calls
`.transform()` on the frozen scaler and `.predict()` on the frozen
KMeans model, then evaluates the quality and internal consistency of
those frozen assignments against the shipped
advanced_persona_profiles.csv dataset.

Random-seed / multi-seed stability analysis is INTENTIONALLY OUT OF
SCOPE here — a deployed model has exactly one seed already baked into
its parameters, so "seed stability" is not a meaningful question to
ask of it. Multi-seed robustness is a separate methodological question
about the *training procedure*, and is covered by the companion
experiment `section_g_clustering_validation.py` (not modified or
re-run by this script).

Outputs
-------
outputs/persona_validation_metrics.csv
outputs/cluster_sizes.csv
outputs/cluster_centroids.csv
outputs/prediction_consistency.csv
outputs/figures/persona_tsne.png
outputs/figures/persona_umap.png        (only if umap-learn is installed)
outputs/figures/persona_cluster_sizes.png
outputs/figures/persona_centroid_heatmap.png
outputs/reports/persona_validation_report.md
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR / "advanced_persona_profiles.csv"
SCALER_PATH = SCRIPT_DIR / "persona_scaler.pkl"
KMEANS_PATH = SCRIPT_DIR / "persona_kmeans.pkl"

OUT_DIR = SCRIPT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"

METRICS_CSV = OUT_DIR / "persona_validation_metrics.csv"
CLUSTER_SIZES_CSV = OUT_DIR / "cluster_sizes.csv"
CLUSTER_CENTROIDS_CSV = OUT_DIR / "cluster_centroids.csv"
PREDICTION_CONSISTENCY_CSV = OUT_DIR / "prediction_consistency.csv"

FIG_TSNE = FIG_DIR / "persona_tsne.png"
FIG_UMAP = FIG_DIR / "persona_umap.png"
FIG_CLUSTER_SIZES = FIG_DIR / "persona_cluster_sizes.png"
FIG_CENTROID_HEATMAP = FIG_DIR / "persona_centroid_heatmap.png"

REPORT_MD = REPORT_DIR / "persona_validation_report.md"

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must exactly match advanced_persona_engine.py feature order
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
    "attention_score",
    "confusion_score",
    "boredom_score",
    "cognitive_overload_score",
]

PERSONA_MAP = {
    0: "Silent Isolator",
    1: "Burnout Pattern",
    2: "Anxiety-Spike Learner",
    3: "Passive Watcher",
    4: "Consistent Learner",
    5: "Last-Minute Survivor",
}

N_CLUSTERS = 6
RANDOM_STATE = 42  # used only for deterministic projections (t-SNE/UMAP), never for the model
SILHOUETTE_SAMPLE_SIZE = 8000  # full-dataset silhouette is O(n^2) and too slow; document the subsample
PROJECTION_SAMPLE_SIZE = 5000

COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#6D6875"]

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("persona_model_validation")


# ─────────────────────────────────────────────────────────────────────────────
# Fail-fast helpers
# ─────────────────────────────────────────────────────────────────────────────
def require_file(path: Path, description: str) -> None:
    if not path.exists():
        log.error("Missing required file: %s (%s)", path, description)
        raise FileNotFoundError(f"Required file not found: {path}")
    log.info("Found %s -> %s", description, path)


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        log.error("Missing required columns in %s: %s", context, missing)
        raise ValueError(f"Missing required columns in {context}: {missing}")


def require_no_fit_capability_misuse(obj, name: str) -> None:
    """
    Sanity guard: confirm the loaded object is a fitted estimator (has the
    sklearn '_fitted' markers) rather than an unfitted stub. This does NOT
    call fit() -- it only checks state that already exists on disk.
    """
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(obj)
    except Exception as exc:  # noqa: BLE001
        log.error("%s failed is-fitted check: %s", name, exc)
        raise RuntimeError(f"{name} does not appear to be a fitted/frozen estimator") from exc
    log.info("%s confirmed fitted/frozen (no fit() will be called).", name)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load frozen artifacts + dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    log.info("=" * 70)
    log.info("STEP 1 — Loading frozen deployed artifacts (joblib, no fit())")
    log.info("=" * 70)

    require_file(SCALER_PATH, "persona_scaler.pkl")
    require_file(KMEANS_PATH, "persona_kmeans.pkl")
    require_file(DATA_PATH, "advanced_persona_profiles.csv")

    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    df = pd.read_csv(DATA_PATH)

    require_no_fit_capability_misuse(scaler, "persona_scaler.pkl")
    require_no_fit_capability_misuse(kmeans, "persona_kmeans.pkl")

    require_columns(df, FEATURES, "advanced_persona_profiles.csv")
    require_columns(df, ["persona_cluster", "persona"], "advanced_persona_profiles.csv")

    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError(
            f"persona_scaler.pkl expects {getattr(scaler, 'n_features_in_', '?')} features, "
            f"but {len(FEATURES)} FEATURES are configured. Refusing to proceed."
        )
    if getattr(kmeans, "n_clusters", None) != N_CLUSTERS:
        raise ValueError(
            f"persona_kmeans.pkl has n_clusters={kmeans.n_clusters}, expected {N_CLUSTERS}."
        )

    if df[FEATURES].isnull().any().any():
        bad = df[FEATURES].isnull().sum()
        bad = bad[bad > 0]
        log.error("Null values found in feature columns:\n%s", bad)
        raise ValueError("advanced_persona_profiles.csv contains null feature values.")

    log.info("Dataset loaded: %s rows x %s cols", *df.shape)
    log.info("Scaler expects %d features: %s", scaler.n_features_in_, list(scaler.feature_names_in_))
    return scaler, kmeans, df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Reproduce frozen predictions (transform + predict only)
# ─────────────────────────────────────────────────────────────────────────────
def reproduce_predictions(scaler, kmeans, df: pd.DataFrame):
    log.info("=" * 70)
    log.info("STEP 2 — Reproducing frozen model predictions (transform/predict only)")
    log.info("=" * 70)

    X_raw = df[FEATURES].copy()
    X_scaled = scaler.transform(X_raw)  # transform() only, never fit_transform()
    predicted_clusters = kmeans.predict(X_scaled)  # predict() only, never fit_predict()

    stored_clusters = df["persona_cluster"].to_numpy()

    match_mask = predicted_clusters == stored_clusters
    n_total = len(df)
    n_match = int(match_mask.sum())
    n_mismatch = n_total - n_match
    match_rate = n_match / n_total

    log.info("Rows total:      %d", n_total)
    log.info("Exact matches:   %d", n_match)
    log.info("Mismatches:      %d", n_mismatch)
    log.info("Match rate:      %.6f%%", match_rate * 100)

    consistency_df = pd.DataFrame(
        {
            "id_student": df.get("id_student", pd.Series(range(n_total))),
            "stored_persona_cluster": stored_clusters,
            "predicted_persona_cluster": predicted_clusters,
            "stored_persona": df["persona"],
            "predicted_persona": [PERSONA_MAP.get(c, f"Unknown-{c}") for c in predicted_clusters],
            "is_match": match_mask,
        }
    )

    if n_mismatch > 0:
        log.warning(
            "%d mismatched rows detected between frozen model predictions and stored "
            "persona_cluster column. These are reported in full in prediction_consistency.csv "
            "and summarized in the validation report.",
            n_mismatch,
        )
    else:
        log.info("PASS — Frozen model predictions exactly reproduce the stored persona_cluster column.")

    return X_scaled, predicted_clusters, consistency_df, {
        "n_total": n_total,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "match_rate": match_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Genuine validation metrics on frozen assignments
# ─────────────────────────────────────────────────────────────────────────────
def compute_validation_metrics(X_scaled: np.ndarray, clusters: np.ndarray) -> dict:
    log.info("=" * 70)
    log.info("STEP 3 — Computing validation metrics on frozen cluster assignments")
    log.info("=" * 70)

    n = X_scaled.shape[0]
    rng = np.random.RandomState(RANDOM_STATE)

    if n > SILHOUETTE_SAMPLE_SIZE:
        sample_idx = rng.choice(n, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        log.info(
            "Dataset has %d rows; silhouette score computed on a %d-row random "
            "subsample (silhouette is O(n^2) and infeasible on the full dataset).",
            n, SILHOUETTE_SAMPLE_SIZE,
        )
    else:
        sample_idx = np.arange(n)

    sil = silhouette_score(X_scaled[sample_idx], clusters[sample_idx])
    db = davies_bouldin_score(X_scaled, clusters)
    ch = calinski_harabasz_score(X_scaled, clusters)

    log.info("Silhouette Score:        %.4f  (n_sample=%d)", sil, len(sample_idx))
    log.info("Davies-Bouldin Index:    %.4f  (full dataset, n=%d)", db, n)
    log.info("Calinski-Harabasz Index: %.4f  (full dataset, n=%d)", ch, n)

    return {
        "silhouette_score": sil,
        "silhouette_sample_size": len(sample_idx),
        "davies_bouldin_index": db,
        "calinski_harabasz_index": ch,
        "n_rows_evaluated": n,
        "n_clusters": N_CLUSTERS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Cluster size / percentage tables
# ─────────────────────────────────────────────────────────────────────────────
def build_cluster_size_table(df: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("STEP 4 — Building cluster size / percentage tables")
    log.info("=" * 70)

    counts = pd.Series(clusters).value_counts().sort_index()
    total = counts.sum()

    table = pd.DataFrame(
        {
            "cluster_id": counts.index,
            "persona": [PERSONA_MAP.get(c, f"Unknown-{c}") for c in counts.index],
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(2),
        }
    )
    log.info("\n%s", table.to_string(index=False))
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Centroid table (from frozen kmeans.cluster_centers_, inverse-scaled)
# ─────────────────────────────────────────────────────────────────────────────
def build_centroid_table(scaler, kmeans) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("STEP 5 — Building centroid table from frozen cluster_centers_")
    log.info("=" * 70)

    centers_scaled = kmeans.cluster_centers_
    # Inverse-transform back to original feature units for interpretability.
    centers_original = scaler.inverse_transform(centers_scaled)

    centroid_df = pd.DataFrame(centers_original, columns=FEATURES)
    centroid_df.insert(0, "persona", [PERSONA_MAP.get(i, f"Unknown-{i}") for i in range(len(centroid_df))])
    centroid_df.insert(0, "cluster_id", range(len(centroid_df)))

    log.info("\n%s", centroid_df.to_string(index=False))
    return centroid_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Figures
# ─────────────────────────────────────────────────────────────────────────────
def generate_tsne_figure(X_scaled: np.ndarray, clusters: np.ndarray, n_rows: int) -> None:
    log.info("Generating t-SNE projection figure ...")
    rng = np.random.RandomState(RANDOM_STATE)
    n = X_scaled.shape[0]
    sample_size = min(PROJECTION_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=sample_size, replace=False)

    tsne = TSNE(n_components=2, perplexity=40, random_state=RANDOM_STATE, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled[idx])
    labels = clusters[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in range(N_CLUSTERS):
        mask = labels == cid
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=COLORS[cid], s=8, alpha=0.6,
            label=PERSONA_MAP.get(cid, f"Cluster {cid}"), edgecolors="none",
        )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title(
        f"Frozen Persona Model — t-SNE Projection\n"
        f"Sample n={sample_size:,} of {n_rows:,} rows  |  k={N_CLUSTERS}  |  perplexity=40",
        fontweight="bold", pad=10,
    )
    ax.legend(markerscale=4, fontsize=9, framealpha=0.9, loc="best", title="Persona")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_TSNE, dpi=300)
    plt.close(fig)
    log.info("[SAVED] %s", FIG_TSNE)


def generate_umap_figure(X_scaled: np.ndarray, clusters: np.ndarray, n_rows: int) -> bool:
    if not HAS_UMAP:
        log.warning(
            "umap-learn is NOT installed in this environment. Skipping UMAP figure "
            "generation. To enable it, run: pip install umap-learn --break-system-packages. "
            "This absence is documented in the validation report."
        )
        return False

    log.info("Generating UMAP projection figure ...")
    rng = np.random.RandomState(RANDOM_STATE)
    n = X_scaled.shape[0]
    sample_size = min(PROJECTION_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=sample_size, replace=False)

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=RANDOM_STATE)
    X_umap = reducer.fit_transform(X_scaled[idx])
    labels = clusters[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in range(N_CLUSTERS):
        mask = labels == cid
        ax.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=COLORS[cid], s=8, alpha=0.6,
            label=PERSONA_MAP.get(cid, f"Cluster {cid}"), edgecolors="none",
        )
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.set_title(
        f"Frozen Persona Model — UMAP Projection\n"
        f"Sample n={sample_size:,} of {n_rows:,} rows  |  k={N_CLUSTERS}  |  n_neighbors=30",
        fontweight="bold", pad=10,
    )
    ax.legend(markerscale=4, fontsize=9, framealpha=0.9, loc="best", title="Persona")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_UMAP, dpi=300)
    plt.close(fig)
    log.info("[SAVED] %s", FIG_UMAP)
    return True


def generate_cluster_size_figure(size_table: pd.DataFrame) -> None:
    log.info("Generating cluster size bar chart ...")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        size_table["persona"], size_table["count"],
        color=COLORS[: len(size_table)], edgecolor="white",
    )
    for bar, pct in zip(bars, size_table["percentage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Student Count", fontsize=11)
    ax.set_title(
        "Frozen Persona Model — Cluster Size Distribution\n"
        f"n={int(size_table['count'].sum()):,} students  |  k={N_CLUSTERS}",
        fontweight="bold", pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(FIG_CLUSTER_SIZES, dpi=300)
    plt.close(fig)
    log.info("[SAVED] %s", FIG_CLUSTER_SIZES)


def generate_centroid_heatmap(centroid_df: pd.DataFrame) -> None:
    log.info("Generating centroid heatmap (matplotlib only, no seaborn) ...")
    feature_cols = FEATURES
    matrix = centroid_df[feature_cols].to_numpy()

    # Per-feature z-normalization purely for heatmap color scaling/readability;
    # the underlying centroid_table.csv retains true original-unit values.
    col_mean = matrix.mean(axis=0)
    col_std = matrix.std(axis=0)
    col_std[col_std == 0] = 1.0
    matrix_norm = (matrix - col_mean) / col_std

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix_norm, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(centroid_df)))
    ax.set_yticklabels(centroid_df["persona"], fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.1f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(matrix_norm[i, j]) > 1 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Z-normalized centroid value (color only)", fontsize=9)

    ax.set_title(
        "Frozen Persona Model — Cluster Centroid Heatmap\n"
        "Cell values are true feature-unit centroids; color is z-normalized per feature",
        fontweight="bold", pad=10,
    )
    plt.tight_layout()
    fig.savefig(FIG_CENTROID_HEATMAP, dpi=300)
    plt.close(fig)
    log.info("[SAVED] %s", FIG_CENTROID_HEATMAP)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Markdown report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(
    dataset_shape: tuple,
    consistency_summary: dict,
    metrics: dict,
    size_table: pd.DataFrame,
    centroid_df: pd.DataFrame,
    umap_generated: bool,
) -> None:
    log.info("=" * 70)
    log.info("STEP 6 — Writing validation report")
    log.info("=" * 70)

    match_rate_pct = consistency_summary["match_rate"] * 100
    consistency_verdict = (
        "PASS — predictions from the frozen model exactly reproduce the stored "
        "`persona_cluster` column."
        if consistency_summary["n_mismatch"] == 0
        else (
            f"ATTENTION — {consistency_summary['n_mismatch']} of "
            f"{consistency_summary['n_total']} rows ({100 - match_rate_pct:.4f}% of the dataset) "
            "do not match the stored `persona_cluster` values. See `prediction_consistency.csv` "
            "for the full row-level mismatch list."
        )
    )

    umap_section = (
        f"![UMAP Projection](../figures/{FIG_UMAP.name})"
        if umap_generated
        else (
            "**UMAP projection was not generated.** The `umap-learn` package is not "
            "installed in the environment this script ran in. Install it with "
            "`pip install umap-learn --break-system-packages` and re-run this script "
            "to produce `persona_umap.png`."
        )
    )

    lines = []
    lines.append("# EduGuard AI — Persona Model External Validation Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "This report validates the **frozen, deployed** persona clustering artifacts "
        "(`persona_scaler.pkl`, `persona_kmeans.pkl`) exactly as they are used in "
        "production by `advanced_persona_engine.py`. No model was trained, fitted, "
        "or re-fitted as part of this validation. Only `.transform()` and `.predict()` "
        "were called on the frozen objects."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Source: `advanced_persona_profiles.csv`")
    lines.append(f"- Shape: {dataset_shape[0]:,} rows x {dataset_shape[1]} columns")
    lines.append(f"- Feature columns used (must match training order exactly): `{', '.join(FEATURES)}`")
    lines.append("")
    lines.append("## 1. Prediction Consistency Check")
    lines.append(
        "The frozen scaler and frozen KMeans model were re-applied to the feature "
        "columns of `advanced_persona_profiles.csv` via `scaler.transform()` followed "
        "by `kmeans.predict()`. The resulting cluster assignments were compared "
        "row-by-row against the `persona_cluster` column already stored in the CSV."
    )
    lines.append("")
    lines.append(f"- Total rows evaluated: **{consistency_summary['n_total']:,}**")
    lines.append(f"- Exact matches: **{consistency_summary['n_match']:,}**")
    lines.append(f"- Mismatches: **{consistency_summary['n_mismatch']:,}**")
    lines.append(f"- Match rate: **{match_rate_pct:.6f}%**")
    lines.append("")
    lines.append(f"**Verdict:** {consistency_verdict}")
    lines.append("")
    lines.append("Full row-level detail: `outputs/prediction_consistency.csv`.")
    lines.append("")
    lines.append("## 2. Validation Metrics (Frozen Cluster Assignments)")
    lines.append(
        "These metrics are computed directly on the frozen model's cluster labels "
        "and the scaler-transformed feature matrix. They measure the *current* "
        "quality of the deployed clustering, not a newly trained one."
    )
    lines.append("")
    lines.append("| Metric | Value | Notes |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Silhouette Score | {metrics['silhouette_score']:.4f} | "
        f"Computed on a random subsample of {metrics['silhouette_sample_size']:,} rows "
        "(silhouette is O(n²); full-dataset computation is infeasible) |"
    )
    lines.append(
        f"| Davies-Bouldin Index | {metrics['davies_bouldin_index']:.4f} | "
        f"Computed on the full dataset (n={metrics['n_rows_evaluated']:,}); lower is better |"
    )
    lines.append(
        f"| Calinski-Harabasz Index | {metrics['calinski_harabasz_index']:.4f} | "
        f"Computed on the full dataset (n={metrics['n_rows_evaluated']:,}); higher is better |"
    )
    lines.append("")
    lines.append("Raw values: `outputs/persona_validation_metrics.csv`.")
    lines.append("")
    lines.append("## 3. Cluster Size Distribution")
    lines.append("")
    lines.append(size_table.to_markdown(index=False))
    lines.append("")
    lines.append("Raw values: `outputs/cluster_sizes.csv`.")
    lines.append("")
    lines.append("![Cluster Size Distribution](../figures/" + FIG_CLUSTER_SIZES.name + ")")
    lines.append("")
    lines.append("## 4. Cluster Centroids")
    lines.append(
        "Centroids below are the frozen `kmeans.cluster_centers_` values, "
        "inverse-transformed back into original feature units via the frozen scaler "
        "for interpretability."
    )
    lines.append("")
    lines.append(centroid_df.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("Raw values: `outputs/cluster_centroids.csv`.")
    lines.append("")
    lines.append("![Centroid Heatmap](../figures/" + FIG_CENTROID_HEATMAP.name + ")")
    lines.append("")
    lines.append("## 5. Cluster Structure Visualization")
    lines.append("")
    lines.append("### t-SNE Projection")
    lines.append("![t-SNE Projection](../figures/" + FIG_TSNE.name + ")")
    lines.append("")
    lines.append("### UMAP Projection")
    lines.append(umap_section)
    lines.append("")
    lines.append("## 6. On Random-Seed Stability (Scope Note)")
    lines.append(
        "This report deliberately does **not** perform multi-seed retraining or "
        "report random-seed stability metrics (e.g., Adjusted Rand Index across seeds). "
        "`persona_kmeans.pkl` is a single, already-fitted, deployed model — it has one "
        "fixed set of cluster centers and was fit with one specific random seed at "
        "training time. Asking \"how stable is this frozen model across different "
        "seeds\" is not a well-formed question, because a frozen model cannot be "
        "re-seeded without becoming a different model."
    )
    lines.append("")
    lines.append(
        "Random-seed / multi-seed robustness is a property of the **training procedure**, "
        "not of a specific deployed artifact. That question is addressed separately by "
        "`section_g_clustering_validation.py`, which re-runs K-Means from scratch across "
        "20 seeds on the training dataset to characterize how sensitive the clustering "
        "*methodology* is to initialization. That script is a distinct, complementary "
        "experiment and is not modified, re-run, or reproduced by this validation script."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Prediction consistency: {consistency_verdict.split(' — ')[0]}"
    )
    lines.append(
        f"- Silhouette / Davies-Bouldin / Calinski-Harabasz computed directly from the "
        "frozen deployed model, no synthetic or fabricated values."
    )
    lines.append(
        "- This validation confirms whether the persona engine **currently deployed** "
        "in EduGuard AI behaves as documented against the shipped reference dataset."
    )
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("[SAVED] %s", REPORT_MD)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("EduGuard AI — Persona Model External Validation")
    log.info("Validating FROZEN deployed artifacts only. No training will occur.")

    scaler, kmeans, df = load_artifacts()

    X_scaled, predicted_clusters, consistency_df, consistency_summary = reproduce_predictions(
        scaler, kmeans, df
    )

    metrics = compute_validation_metrics(X_scaled, predicted_clusters)

    size_table = build_cluster_size_table(df, predicted_clusters)
    centroid_df = build_centroid_table(scaler, kmeans)

    log.info("=" * 70)
    log.info("Writing output tables")
    log.info("=" * 70)
    pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)
    log.info("[SAVED] %s", METRICS_CSV)

    size_table.to_csv(CLUSTER_SIZES_CSV, index=False)
    log.info("[SAVED] %s", CLUSTER_SIZES_CSV)

    centroid_df.to_csv(CLUSTER_CENTROIDS_CSV, index=False)
    log.info("[SAVED] %s", CLUSTER_CENTROIDS_CSV)

    consistency_df.to_csv(PREDICTION_CONSISTENCY_CSV, index=False)
    log.info("[SAVED] %s", PREDICTION_CONSISTENCY_CSV)

    log.info("=" * 70)
    log.info("Generating figures")
    log.info("=" * 70)
    generate_tsne_figure(X_scaled, predicted_clusters, n_rows=len(df))
    umap_generated = generate_umap_figure(X_scaled, predicted_clusters, n_rows=len(df))
    generate_cluster_size_figure(size_table)
    generate_centroid_heatmap(centroid_df)

    write_report(
        dataset_shape=df.shape,
        consistency_summary=consistency_summary,
        metrics=metrics,
        size_table=size_table,
        centroid_df=centroid_df,
        umap_generated=umap_generated,
    )

    log.info("=" * 70)
    log.info("VALIDATION COMPLETE")
    log.info("=" * 70)
    log.info("Prediction match rate: %.6f%%", consistency_summary["match_rate"] * 100)
    log.info("Silhouette Score:      %.4f", metrics["silhouette_score"])
    log.info("Davies-Bouldin Index:  %.4f", metrics["davies_bouldin_index"])
    log.info("Calinski-Harabasz:     %.4f", metrics["calinski_harabasz_index"])
    if consistency_summary["n_mismatch"] > 0:
        log.warning(
            "%d mismatches detected — review prediction_consistency.csv and the report.",
            consistency_summary["n_mismatch"],
        )


if __name__ == "__main__":
    main()
