"""
EduGuard AI — Compute Missing Metrics
======================================
1. Calinski-Harabasz Index for KMeans clustering
2. Memory / CPU usage for deployment benchmark
3. Inference time per model (for deep learning comparison table)

Outputs:
  ieee_results/calinski_harabasz.json
  ieee_results/deployment_full_benchmark.json
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

# ===========================================================================
# PART 1 — Calinski-Harabasz + Full Clustering Metrics
# ===========================================================================
print("=" * 60)
print("PART 1 — Calinski-Harabasz Index")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)

df     = pd.read_csv(DATA_PATH)
X_raw  = df[FEATURES].copy()
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw)

# Use same params as section_g
km = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
labels = km.fit_predict(X_sc)

# Subsample for silhouette (matching section_g)
rng        = np.random.RandomState(42)
sample_idx = rng.choice(len(X_sc), size=8000, replace=False)

sil = silhouette_score(X_sc[sample_idx], labels[sample_idx])
db  = davies_bouldin_score(X_sc, labels)
ch  = calinski_harabasz_score(X_sc, labels)

print(f"  Silhouette Score     : {sil:.4f}")
print(f"  Davies-Bouldin Index : {db:.4f}")
print(f"  Calinski-Harabasz    : {ch:.2f}")
print(f"  Inertia (WCSS)       : {km.inertia_:.2f}")

cluster_metrics = {
    "n_clusters":          6,
    "silhouette_score":    round(float(sil), 4),
    "davies_bouldin":      round(float(db),  4),
    "calinski_harabasz":   round(float(ch),  2),
    "inertia":             round(float(km.inertia_), 2),
    "cluster_sizes":       pd.Series(labels).value_counts().sort_index().tolist(),
}

with open(os.path.join(OUT_DIR, "calinski_harabasz.json"), "w") as f:
    json.dump(cluster_metrics, f, indent=2)
print("  [SAVED] calinski_harabasz.json")

# ===========================================================================
# PART 2 — Deployment Benchmark (Memory + CPU + Latency)
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2 — Deployment Benchmark (Memory + CPU + Inference)")
print("=" * 60)

import joblib
from xgboost import XGBClassifier

MODEL_PATH = os.path.join(SCRIPT_DIR, "academic_risk_xgboost.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("  [OK] XGBoost model loaded from pkl")
except Exception:
    print("  [WARN] Retraining XGBoost for benchmark...")
    from sklearn.model_selection import train_test_split
    X = df[FEATURES].values
    y = df["psychological_risk"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0)
    model.fit(X_train, y_train)

# Warm-up
X_bench = df[FEATURES].values[:1]
_ = model.predict(X_bench)

# Memory baseline
proc      = psutil.Process(os.getpid())
mem_mb_base = proc.memory_info().rss / (1024 ** 2)

# Run inference benchmarks
batch_sizes = [1, 10, 100, 1000, 10000]
benchmark_rows = []

for bs in batch_sizes:
    X_b = df[FEATURES].values[:bs]
    N_REPS = max(5, 200 // bs)

    latencies = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        model.predict_proba(X_b)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    lat_mean   = float(np.mean(latencies))
    lat_std    = float(np.std(latencies))
    per_req    = lat_mean / bs
    throughput = 1000.0 / per_req  # requests per second

    # Memory after inference
    mem_after = proc.memory_info().rss / (1024 ** 2)
    mem_delta = mem_after - mem_mb_base

    cpu_pct   = psutil.cpu_percent(interval=None)

    benchmark_rows.append({
        "batch_size":         bs,
        "latency_mean_ms":    round(lat_mean,   4),
        "latency_std_ms":     round(lat_std,    4),
        "per_request_ms":     round(per_req,    6),
        "throughput_req_s":   round(throughput, 1),
        "memory_usage_mb":    round(mem_after,  1),
        "memory_delta_mb":    round(mem_delta,  1),
        "cpu_pct":            round(cpu_pct,    1),
    })

    print(f"  batch={bs:6d}  lat={lat_mean:.3f}ms±{lat_std:.3f}  "
          f"per_req={per_req:.4f}ms  tput={throughput:.0f}req/s  "
          f"mem={mem_after:.0f}MB  cpu={cpu_pct:.1f}%")

# Model file size
model_size_kb = os.path.getsize(MODEL_PATH) / 1024 if os.path.exists(MODEL_PATH) else 0
print(f"\n  Model file size: {model_size_kb:.1f} KB")

full_benchmark = {
    "model": "XGBoost",
    "model_size_kb": round(model_size_kb, 1),
    "benchmark": benchmark_rows,
}

with open(os.path.join(OUT_DIR, "deployment_full_benchmark.json"), "w") as f:
    json.dump(full_benchmark, f, indent=2)
print("  [SAVED] deployment_full_benchmark.json")

# ===========================================================================
# PART 3 — Summary Print
# ===========================================================================
print("\n" + "=" * 60)
print("  ALL MISSING METRICS COMPUTED SUCCESSFULLY")
print("=" * 60)
print(f"\n  Calinski-Harabasz Index : {ch:.2f}")
print(f"  Model file size         : {model_size_kb:.1f} KB")
print(f"  Batch-1 latency         : {benchmark_rows[0]['latency_mean_ms']:.3f} ms")
print(f"  Batch-1 memory          : {benchmark_rows[0]['memory_usage_mb']:.0f} MB")
print("\n  Done.\n")
