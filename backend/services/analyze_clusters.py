"""
Cluster analysis script — find what feature values correspond to each cluster,
and find the optimal seed values to guarantee correct persona prediction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "../../ml/persona_kmeans.pkl")
SCALER_PATH = os.path.join(BASE, "../../ml/persona_scaler.pkl")

persona_model = joblib.load(MODEL_PATH)
persona_scaler = joblib.load(SCALER_PATH)

persona_map = {
    0: "Silent Isolator",
    1: "Burnout Pattern",
    2: "Anxiety-Spike Learner",
    3: "Passive Watcher",
    4: "Consistent Learner",
    5: "Last-Minute Survivor"
}

feature_names = [
    "total_clicks", "avg_score", "active_days", "engagement_variability",
    "inactivity_days", "engagement_slope", "assessment_consistency",
    "attention_score", "confusion_score", "boredom_score", "cognitive_overload_score"
]

print("=== CLUSTER CENTERS (in original feature space) ===")
centers_scaled = persona_model.cluster_centers_
try:
    centers_original = persona_scaler.inverse_transform(centers_scaled)
    for cluster_id, center in enumerate(centers_original):
        persona = persona_map[cluster_id]
        print(f"\nCluster {cluster_id} — {persona}:")
        for fname, fval in zip(feature_names, center):
            print(f"  {fname}: {fval:.2f}")
except Exception as e:
    print(f"Could not inverse transform: {e}")
    print("\nScaled centers:")
    for cluster_id, center in enumerate(centers_scaled):
        print(f"  Cluster {cluster_id} — {persona_map[cluster_id]}: {center}")

print("\n=== FINDING ROBUST SEEDS FOR EACH TARGET CLUSTER ===")

def predict_persona(features_dict):
    df = pd.DataFrame([features_dict])
    scaled = persona_scaler.transform(df)
    cluster = int(persona_model.predict(scaled)[0])
    return cluster, persona_map.get(cluster, "Unknown")

# Try variations around each cluster center to find guaranteed seeds
for target_cluster in [4, 0, 1]:  # Consistent, Silent Isolator, Burnout for scenarios
    center_scaled = centers_scaled[target_cluster]
    center_original = persona_scaler.inverse_transform([center_scaled])[0]
    features = dict(zip(feature_names, center_original))

    cluster_pred, persona_pred = predict_persona(features)
    print(f"\nCluster center for {persona_map[target_cluster]} (cluster {target_cluster}):")
    print(f"  Predicted as: {persona_pred} (cluster={cluster_pred}) — {'CORRECT' if cluster_pred == target_cluster else 'WRONG!'}")
    for k, v in features.items():
        print(f"    {k}: {v:.2f}")
