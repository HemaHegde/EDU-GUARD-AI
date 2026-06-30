import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.metrics import adjusted_rand_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

print("=" * 60)
print("Evaluating K-Means with k=4 for Comparison")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X_raw = df[FEATURES].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)

# 1. Main Clustering at k=4
km = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
labels = km.fit_predict(X_sc)

# Metrics
# Using a sample of 8000 for silhouette to match what was done for k=6
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_sc), size=8000, replace=False)

sil = silhouette_score(X_sc[sample_idx], labels[sample_idx])
db  = davies_bouldin_score(X_sc, labels)
ch  = calinski_harabasz_score(X_sc, labels)
inertia = km.inertia_

print(f"  Silhouette Score     : {sil:.4f}")
print(f"  Davies-Bouldin Index : {db:.4f}")
print(f"  Calinski-Harabasz    : {ch:.2f}")
print(f"  Inertia (WCSS)       : {inertia:.2f}")
print(f"  Cluster sizes        : {pd.Series(labels).value_counts().sort_index().tolist()}")

# 2. Stability Testing at k=4 (ARI over 10 random seeds to match methodology)
# (In the paper they used 20 seeds, we'll do 10 to be fast, or 20 if we want to be exact)
print("\nComputing ARI stability over 20 random seeds for k=4...")
seeds = range(100, 120)
seed_labels = []
for s in seeds:
    km_tmp = KMeans(n_clusters=4, random_state=s, n_init=10, max_iter=300)
    seed_labels.append(km_tmp.fit_predict(X_sc))

aris = []
for i in range(len(seeds)):
    for j in range(i + 1, len(seeds)):
        aris.append(adjusted_rand_score(seed_labels[i], seed_labels[j]))

ari_mean = np.mean(aris)
ari_std = np.std(aris)

print(f"  ARI Cluster Stability: {ari_mean:.4f} ± {ari_std:.4f}")
print("\nDone.")
