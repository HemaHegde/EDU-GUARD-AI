"""
Quick validation script to verify that the KMeans persona model correctly
classifies each evaluation seed profile into the expected persona.

Run from the backend directory with the venv Python:
  python validate_eval_personas.py
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

# Load models
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

# Seed profiles (from seed_service.py)
DEMO_PROFILES = {
    "consistent": {
        "label": "Consistent Learner",
        "features": {
            "total_clicks": 847,
            "avg_score": 78.3,
            "active_days": 26,
            "engagement_variability": 2.8,
            "inactivity_days": 4,
            "engagement_slope": 0.15,
            "assessment_consistency": 4.2,
        },
        "cognitive": {
            "focus_score": 82,
            "engagement_score": 88,
            "confusion_score": 18,
            "learning_velocity": 76,
            "ai_risk_probability": 0.12,
        },
    },
    "last_minute": {
        "label": "Last-Minute Survivor",
        "features": {
            "total_clicks": 2275,
            "avg_score": 85.6,
            "active_days": 78,
            "engagement_variability": 20.51,
            "inactivity_days": 160,
            "engagement_slope": -0.21,
            "assessment_consistency": 10.9,
        },
        "cognitive": {
            "focus_score": 85,
            "engagement_score": 100,
            "confusion_score": 14,
            "learning_velocity": 65,
            "ai_risk_probability": 0.68,
        },
    },
    "burnout": {
        "label": "Burnout Pattern",
        "features": {
            "total_clicks": 87,
            "avg_score": 41.2,
            "active_days": 8,
            "engagement_variability": 15.3,
            "inactivity_days": 6,
            "engagement_slope": -0.24,
            "assessment_consistency": 18.9,
        },
        "cognitive": {
            "focus_score": 34,
            "engagement_score": 28,
            "confusion_score": 74,
            "learning_velocity": 22,
            "ai_risk_probability": 0.89,
        },
    },
}

EVAL_SCENARIOS = {
    "low":      {"profile": "consistent",   "expected_persona": "Consistent Learner"},
    "moderate": {"profile": "last_minute",  "expected_persona": "Last-Minute Survivor"},
    "high":     {"profile": "burnout",      "expected_persona": "Burnout Pattern"},
}

print("=" * 60)
print("IEEE EVAL PERSONA VALIDATION")
print("=" * 60)

all_ok = True

for scenario_name, scenario in EVAL_SCENARIOS.items():
    profile_key = scenario["profile"]
    expected = scenario["expected_persona"]
    profile = DEMO_PROFILES[profile_key]

    f = profile["features"]
    c = profile["cognitive"]

    total_clicks = float(f.get("total_clicks", 0) or 0)
    avg_score = float(f.get("avg_score", 0) or 0)
    active_days = float(f.get("active_days", 0) or 0)
    engagement_variability = float(f.get("engagement_variability", 0) or 0)
    inactivity_days = float(f.get("inactivity_days", 0) or 0)
    engagement_slope = float(f.get("engagement_slope", 0) or 0)
    assessment_consistency = float(f.get("assessment_consistency", 0) or 0)

    focus_score = float(c.get("focus_score", 50) or 50)
    confusion_score = float(c.get("confusion_score", 50) or 50)
    engagement_score = float(c.get("engagement_score", 50) or 50)
    ai_risk_probability = float(c.get("ai_risk_probability", 0) or 0)

    if ai_risk_probability <= 1:
        cognitive_overload_score = ai_risk_probability * 100
    else:
        cognitive_overload_score = ai_risk_probability

    attention_score = focus_score
    boredom_score = max(0, 100 - engagement_score)

    model_input = pd.DataFrame([{
        "total_clicks": total_clicks,
        "avg_score": avg_score,
        "active_days": active_days,
        "engagement_variability": engagement_variability,
        "inactivity_days": inactivity_days,
        "engagement_slope": engagement_slope,
        "assessment_consistency": assessment_consistency,
        "attention_score": attention_score,
        "confusion_score": confusion_score,
        "boredom_score": boredom_score,
        "cognitive_overload_score": cognitive_overload_score,
    }])

    scaled = persona_scaler.transform(model_input)
    cluster = int(persona_model.predict(scaled)[0])
    predicted_persona = persona_map.get(cluster, "Unknown")

    match = predicted_persona == expected
    status = "✓ PASS" if match else "✗ FAIL"
    all_ok = all_ok and match

    print(f"\nScenario: {scenario_name.upper()}")
    print(f"  Seed Profile  : {profile_key}")
    print(f"  Expected      : {expected}")
    print(f"  Predicted     : {predicted_persona} (cluster={cluster})")
    print(f"  Status        : {status}")

print("\n" + "=" * 60)
if all_ok:
    print("ALL PERSONAS MATCH — no mismatches detected")
else:
    print("WARNING: PERSONA MISMATCHES DETECTED — see above")
print("=" * 60)
