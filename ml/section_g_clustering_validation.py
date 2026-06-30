"""
EduGuard AI — Section G: Clustering Validation (Multi-Seed)
============================================================
Runs K-Means with 20 different random seeds.

Reports:
  Silhouette Score  — Mean ± SD
  Davies-Bouldin    — Mean ± SD
  Cluster Stability — Adjusted Rand Index between runs

Generates:
  PCA visualization
  t-SNE visualization
  UMAP visualization (if umap-learn installed)

Outputs:
  ieee_results/section_g_clustering_validation.csv
  ieee_results/section_g_clustering_validation.json
  ieee_results/figures/fig_g1_pca_clusters.png
  ieee_results/figures/fig_g2_tsne_clusters.png
  ieee_results/figures/fig_g3_umap_clusters.png  (if umap-learn installed)
  ieee_results/figures/fig_g4_silhouette_stability.png
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score
)

warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed. Run: pip install umap-learn")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]
N_CLUSTERS = 6
PERSONAS   = [
    "Silent Isolator", "Burnout Pattern", "Anxiety-Spike Learner",
    "Passive Watcher", "Consistent Learner", "Last-Minute Survivor"
]

# ── Load & scale ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Section G — Clustering Validation (20 seeds)")
print("=" * 60)

df      = pd.read_csv(DATA_PATH)
X_raw   = df[FEATURES].copy()
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_raw)
print(f"  Dataset: {len(df):,} rows  |  Features: {len(FEATURES)}")

# ── Subsample for silhouette (full 32k is slow) ────────────────────────────────
rng         = np.random.RandomState(42)
sample_idx  = rng.choice(len(X_sc), size=8000, replace=False)
X_sample    = X_sc[sample_idx]

# ── Multi-seed K-Means ─────────────────────────────────────────────────────────
print(f"\n  Running K-Means with 20 different seeds (k={N_CLUSTERS}) ...")

seeds       = list(range(0, 20))
sil_scores  = []
db_scores   = []
all_labels  = []

for seed in seeds:
    km = KMeans(n_clusters=N_CLUSTERS, random_state=seed,
                n_init=10, max_iter=300)
    labels = km.fit_predict(X_sc)
    all_labels.append(labels)

    sil = silhouette_score(X_sample, labels[sample_idx])
    db  = davies_bouldin_score(X_sc, labels)
    sil_scores.append(sil)
    db_scores.append(db)
    print(f"    Seed {seed:02d}: Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")

sil_mean = np.mean(sil_scores)
sil_std  = np.std(sil_scores)
db_mean  = np.mean(db_scores)
db_std   = np.std(db_scores)

print(f"\n  Silhouette Score:    {sil_mean:.4f} ± {sil_std:.4f}")
print(f"  Davies-Bouldin:      {db_mean:.4f} ± {db_std:.4f}")

# ── Cluster Stability (ARI between all pairs) ──────────────────────────────────
print("\n  Computing Cluster Stability (Adjusted Rand Index) ...")
ari_scores = []
for i in range(len(seeds)):
    for j in range(i+1, len(seeds)):
        ari = adjusted_rand_score(all_labels[i], all_labels[j])
        ari_scores.append(ari)

ari_mean = np.mean(ari_scores)
ari_std  = np.std(ari_scores)
print(f"  Cluster Stability (ARI mean):  {ari_mean:.4f} ± {ari_std:.4f}")

# ── Use seed=42 labels for visualization ──────────────────────────────────────
km_best  = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20, max_iter=500)
labels42 = km_best.fit_predict(X_sc)

# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "n_clusters":          N_CLUSTERS,
    "n_seeds":             len(seeds),
    "silhouette_mean":     round(sil_mean, 4),
    "silhouette_std":      round(sil_std,  4),
    "davies_bouldin_mean": round(db_mean,  4),
    "davies_bouldin_std":  round(db_std,   4),
    "cluster_stability_ari_mean": round(ari_mean, 4),
    "cluster_stability_ari_std":  round(ari_std,  4),
    "per_seed_silhouette":        [round(s, 4) for s in sil_scores],
    "per_seed_davies_bouldin":    [round(d, 4) for d in db_scores],
}

with open(os.path.join(OUT_DIR, "section_g_clustering_validation.json"), "w") as f:
    json.dump(results, f, indent=2)

seed_df = pd.DataFrame({
    "seed": seeds,
    "silhouette": sil_scores,
    "davies_bouldin": db_scores,
})
seed_df.to_csv(os.path.join(OUT_DIR, "section_g_clustering_validation.csv"), index=False)
print("  [SAVED] section_g_clustering_validation.csv")

# ── Plot colors ────────────────────────────────────────────────────────────────
COLORS = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261","#6D6875"]

# ── Fig G1: PCA Visualization ─────────────────────────────────────────────────
print("\n  Generating PCA visualization ...")
pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_sc)
var_exp = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))
for cid in range(N_CLUSTERS):
    mask = labels42 == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[cid], s=6, alpha=0.5, label=PERSONAS[cid],
               edgecolors="none")
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("Figure G1: PCA Visualization of K-Means Learner Personas\n"
             f"OULAD Dataset (n={len(df):,})  k={N_CLUSTERS}  Silhouette={sil_mean:.3f}±{sil_std:.3f}",
             fontweight="bold", pad=10)
leg = ax.legend(markerscale=4, fontsize=9, framealpha=0.9,
                loc="upper right", title="Persona")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_g1_pca_clusters.png"), dpi=300)
plt.close()
print("  [SAVED] fig_g1_pca_clusters.png")

# ── Fig G2: t-SNE Visualization ───────────────────────────────────────────────
print("  Generating t-SNE visualization (may take ~2 min) ...")
# Use 5000-sample subset for speed
tsne_idx = rng.choice(len(X_sc), size=5000, replace=False)
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
            random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_sc[tsne_idx])
l_tsne = labels42[tsne_idx]

fig, ax = plt.subplots(figsize=(9, 7))
for cid in range(N_CLUSTERS):
    mask = l_tsne == cid
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=COLORS[cid], s=8, alpha=0.6, label=PERSONAS[cid],
               edgecolors="none")
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.set_title("Figure G2: t-SNE Visualization of Learner Persona Clusters\n"
             f"OULAD (5,000 sample)  perplexity=40  k={N_CLUSTERS}",
             fontweight="bold", pad=10)
ax.legend(markerscale=4, fontsize=9, framealpha=0.9,
          loc="best", title="Persona")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_g2_tsne_clusters.png"), dpi=300)
plt.close()
print("  [SAVED] fig_g2_tsne_clusters.png")

# ── Fig G3: UMAP ──────────────────────────────────────────────────────────────
if HAS_UMAP:
    print("  Generating UMAP visualization ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30,
                        min_dist=0.1, random_state=42)
    X_umap  = reducer.fit_transform(X_sc[tsne_idx])

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in range(N_CLUSTERS):
        mask = l_tsne == cid
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=COLORS[cid], s=8, alpha=0.6, label=PERSONAS[cid],
                   edgecolors="none")
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.set_title("Figure G3: UMAP Visualization of Learner Persona Clusters\n"
                 f"OULAD (5,000 sample)  n_neighbors=30  k={N_CLUSTERS}",
                 fontweight="bold", pad=10)
    ax.legend(markerscale=4, fontsize=9, framealpha=0.9, title="Persona")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_g3_umap_clusters.png"), dpi=300)
    plt.close()
    print("  [SAVED] fig_g3_umap_clusters.png")

# ── Fig G4: Silhouette stability across seeds ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(seeds, sil_scores, color="#457B9D", edgecolor="white")
axes[0].axhline(sil_mean, color="red", linestyle="--", lw=1.5,
                label=f"Mean = {sil_mean:.4f}")
axes[0].fill_between([-0.5, 19.5],
                     sil_mean - sil_std, sil_mean + sil_std,
                     alpha=0.15, color="red", label=f"±SD = {sil_std:.4f}")
axes[0].set_xlabel("Random Seed", fontsize=10)
axes[0].set_ylabel("Silhouette Score", fontsize=10)
axes[0].set_title("Silhouette Score — 20 Seeds", fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].set_xticks(seeds)
axes[0].grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].bar(seeds, db_scores, color="#E63946", edgecolor="white")
axes[1].axhline(db_mean, color="navy", linestyle="--", lw=1.5,
                label=f"Mean = {db_mean:.4f}")
axes[1].fill_between([-0.5, 19.5],
                     db_mean - db_std, db_mean + db_std,
                     alpha=0.15, color="navy", label=f"±SD = {db_std:.4f}")
axes[1].set_xlabel("Random Seed", fontsize=10)
axes[1].set_ylabel("Davies-Bouldin Index", fontsize=10)
axes[1].set_title("Davies-Bouldin Index — 20 Seeds", fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].set_xticks(seeds)
axes[1].grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle(f"Figure G4: K-Means Clustering Stability (k={N_CLUSTERS}, 20 Seeds)\n"
             f"ARI Stability Score = {ari_mean:.4f} ± {ari_std:.4f}",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_g4_silhouette_stability.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [SAVED] fig_g4_silhouette_stability.png")

print(f"\n  ══ FINAL CLUSTERING METRICS ══")
print(f"  Silhouette Score:  {sil_mean:.4f} ± {sil_std:.4f}")
print(f"  Davies-Bouldin:    {db_mean:.4f} ± {db_std:.4f}")
print(f"  Cluster Stability: {ari_mean:.4f} ± {ari_std:.4f}")
print("\n  Section G complete.\n")
