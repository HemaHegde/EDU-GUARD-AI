"""
Quick check: what risk level does XGBoost predict for each eval profile?
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "../ml/academic_risk_xgboost.pkl"))

PROFILES = {
    "consistent (LR-001)": {
        "total_clicks": 5264, "avg_score": 81.0, "active_days": 169,
        "engagement_variability": 8.6, "inactivity_days": 96,
        "engagement_slope": -0.02, "assessment_consistency": 10.2,
    },
    "last_minute (MR-002)": {
        "total_clicks": 2275, "avg_score": 85.6, "active_days": 78,
        "engagement_variability": 20.51, "inactivity_days": 160,
        "engagement_slope": -0.21, "assessment_consistency": 10.9,
    },
    "burnout (HR-003)": {
        "total_clicks": 117, "avg_score": 15.6, "active_days": 7,
        "engagement_variability": 2.2, "inactivity_days": 20,
        "engagement_slope": -0.17, "assessment_consistency": 0.31,
    },
}

RISK_MAP = {
    "High": "Needs Support (High)",
    "Medium": "On Track (Medium)",
    "Low": "Doing Well (Low)",
}

print("=" * 60)
print("XGBoost Risk Level Predictions for Eval Profiles")
print("=" * 60)
for name, feat in PROFILES.items():
    df = pd.DataFrame([feat])
    prob = float(model.predict_proba(df)[0][1])
    score = int(prob * 100)
    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"
    else:
        level = "Low"
    print(f"\n{name}:")
    print(f"  Risk Probability : {prob:.4f}")
    print(f"  Risk Score       : {score}/100")
    print(f"  Risk Level       : {level}")
    print(f"  UI Display       : {RISK_MAP[level]}")
