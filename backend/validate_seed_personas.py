"""
Quick validation: check that seed_service.py profiles produce correct KMeans personas.
Run from d:\\EDU AI\\edu-ai\\backend directory with venv python.
"""
import sys, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "../ml/persona_kmeans.pkl")
SCALER_PATH = os.path.join(BASE, "../ml/persona_scaler.pkl")

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

# Profiles from seed_service.py (what the frontend eval mode actually uses)
SEED_PROFILES = {
    "consistent": {
        "expected": "Consistent Learner",
        "seed": "S-LR-001",
        "scenario": "low",
        "features": {
            "total_clicks": 5264, "avg_score": 81.0, "active_days": 169,
            "engagement_variability": 8.6, "inactivity_days": 96,
            "engagement_slope": -0.02, "assessment_consistency": 10.2,
        },
        "cognitive": {
            "focus_score": 82, "engagement_score": 90, "confusion_score": 18,
            "learning_velocity": 76, "ai_risk_probability": 0.516,
        },
    },
    "last_minute": {
        "expected": "Last-Minute Survivor",
        "seed": "S-MR-002",
        "scenario": "moderate",
        "features": {
            "total_clicks": 2275, "avg_score": 85.6, "active_days": 78,
            "engagement_variability": 20.51, "inactivity_days": 160,
            "engagement_slope": -0.21, "assessment_consistency": 10.9,
        },
        "cognitive": {
            "focus_score": 85, "engagement_score": 100, "confusion_score": 14,
            "learning_velocity": 65, "ai_risk_probability": 0.68,
        },
    },
    "burnout": {
        "expected": "Burnout Pattern",
        "seed": "S-HR-003",
        "scenario": "high",
        "features": {
            "total_clicks": 117, "avg_score": 15.6, "active_days": 7,
            "engagement_variability": 2.2, "inactivity_days": 20,
            "engagement_slope": -0.17, "assessment_consistency": 0.31,
        },
        "cognitive": {
            "focus_score": 86, "engagement_score": 91, "confusion_score": 14,
            "learning_velocity": 22, "ai_risk_probability": 0.485,
        },
    },
}

print("=" * 60)
print("SEED_SERVICE.PY PERSONA VALIDATION (What frontend uses in eval mode)")
print("=" * 60)

all_ok = True

for profile_key, profile in SEED_PROFILES.items():
    f = profile["features"]
    c = profile["cognitive"]

    focus_score = float(c.get("focus_score", 50))
    engagement_score = float(c.get("engagement_score", 50))
    confusion_score = float(c.get("confusion_score", 50))
    ai_risk_probability = float(c.get("ai_risk_probability", 0))

    if ai_risk_probability <= 1:
        cognitive_overload_score = ai_risk_probability * 100
    else:
        cognitive_overload_score = ai_risk_probability

    attention_score = focus_score
    boredom_score = max(0, 100 - engagement_score)

    model_input = pd.DataFrame([{
        "total_clicks":           float(f["total_clicks"]),
        "avg_score":              float(f["avg_score"]),
        "active_days":            float(f["active_days"]),
        "engagement_variability": float(f["engagement_variability"]),
        "inactivity_days":        float(f["inactivity_days"]),
        "engagement_slope":       float(f["engagement_slope"]),
        "assessment_consistency": float(f["assessment_consistency"]),
        "attention_score":        attention_score,
        "confusion_score":        confusion_score,
        "boredom_score":          boredom_score,
        "cognitive_overload_score": cognitive_overload_score,
    }])

    scaled = persona_scaler.transform(model_input)
    cluster = int(persona_model.predict(scaled)[0])
    predicted_persona = persona_map.get(cluster, "Unknown")

    match = predicted_persona == profile["expected"]
    status = "PASS" if match else "FAIL"
    all_ok = all_ok and match

    print(f"\nScenario {profile['scenario'].upper()} ({profile['seed']}):")
    print(f"  Profile Key   : {profile_key}")
    print(f"  Expected      : {profile['expected']}")
    print(f"  Predicted     : {predicted_persona} (cluster={cluster})")
    print(f"  Status        : {status}")
    if not match:
        print(f"  !! MISMATCH: frontend will throw SCENARIO MISMATCH error !!")

print("\n" + "=" * 60)
if all_ok:
    print("ALL PERSONAS MATCH -- no mismatches detected")
else:
    print("WARNING: PERSONA MISMATCHES DETECTED -- see above")
print("=" * 60)
