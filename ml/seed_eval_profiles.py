"""
IEEE Evaluation Screenshot Seeder
===================================
Seeds the EXACT feature values from human_eval_cases.json
into the Supabase database for the logged-in evaluation user.

Run AFTER starting the backend. The user_id comes from the
Supabase auth session of the currently logged-in user.
"""

import sys, io, json, random
from datetime import datetime, timedelta

sys.path.insert(0, r"D:\EDU AI\edu-ai\backend")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config.supabase_client import supabase

# ─── Load real extraction data ────────────────────────────────────────────────
with open(r"D:\EDU AI\edu-ai\ml\ieee_results\human_eval_cases.json", "r", encoding="utf-8") as f:
    CASES = json.load(f)

# ─── Evaluation profiles matching our 3 real students ─────────────────────────
EVAL_PROFILES = {
    "consistent": {
        "label":    "S-LR-001 — Low Risk (Consistent Learner)",
        "features": CASES["Low Risk"]["features"]["raw"],
        "cognitive": {
            "focus_score":        int(CASES["Low Risk"]["cognitive_state"]["data"]["attention_score"]),
            "engagement_score":   int(100 - CASES["Low Risk"]["cognitive_state"]["data"]["boredom_score"]),
            "confusion_score":    int(CASES["Low Risk"]["cognitive_state"]["data"]["confusion_score"]),
            "learning_velocity":  int(CASES["Low Risk"]["cognitive_state"]["data"]["stability_score"]),
            "ai_risk_probability": CASES["Low Risk"]["risk"]["predicted_probability"],
        },
    },
    "last_minute": {
        "label":    "S-MR-002 — Moderate Risk (Last-Minute Survivor)",
        "features": CASES["Moderate Risk"]["features"]["raw"],
        "cognitive": {
            "focus_score":        int(CASES["Moderate Risk"]["cognitive_state"]["data"]["attention_score"]),
            "engagement_score":   int(100 - CASES["Moderate Risk"]["cognitive_state"]["data"]["boredom_score"]),
            "confusion_score":    int(CASES["Moderate Risk"]["cognitive_state"]["data"]["confusion_score"]),
            "learning_velocity":  int(CASES["Moderate Risk"]["cognitive_state"]["data"]["stability_score"]),
            "ai_risk_probability": CASES["Moderate Risk"]["risk"]["predicted_probability"],
        },
    },
    "burnout": {
        "label":    "S-HR-003 — High Risk (Burnout Pattern)",
        "features": CASES["High Risk"]["features"]["raw"],
        "cognitive": {
            "focus_score":        int(CASES["High Risk"]["cognitive_state"]["data"]["attention_score"]),
            "engagement_score":   int(100 - CASES["High Risk"]["cognitive_state"]["data"]["boredom_score"]),
            "confusion_score":    int(CASES["High Risk"]["cognitive_state"]["data"]["confusion_score"]),
            "learning_velocity":  int(CASES["High Risk"]["cognitive_state"]["data"]["stability_score"]),
            "ai_risk_probability": CASES["High Risk"]["risk"]["predicted_probability"],
        },
    },
}


def seed_eval_profile(user_id: str, profile_key: str):
    profile = EVAL_PROFILES[profile_key]
    print(f"\n  Seeding: {profile['label']}")

    # 1. Verify user in profiles table
    resp = supabase.table("profiles").select("id, full_name").eq("id", user_id).execute()
    if not resp.data:
        print(f"    ERROR: user_id={user_id} not found in profiles table.")
        return False
    student_name = resp.data[0].get("full_name", "Evaluation Student")
    print(f"    Student name: {student_name}")

    # 2. Seed student_features (exact values from pipeline)
    feat = dict(profile["features"])
    # Round all floats to reasonable precision for DB
    feat_record = {
        "user_id":                user_id,
        "total_clicks":           int(feat["total_clicks"]),
        "avg_score":              round(feat["avg_score"], 4),
        "active_days":            int(feat["active_days"]),
        "engagement_variability": round(feat["engagement_variability"], 6),
        "inactivity_days":        int(feat["inactivity_days"]),
        "engagement_slope":       round(feat["engagement_slope"], 8),
        "assessment_consistency": round(feat["assessment_consistency"], 6),
        "updated_at":             datetime.utcnow().isoformat(),
    }
    supabase.table("student_features").upsert(feat_record, on_conflict="user_id").execute()
    print(f"    student_features seeded: {feat_record}")

    # 3. Seed student_activity (simulated from active_days)
    now = datetime.utcnow()
    activity_records = []
    active_days = int(feat["active_days"])
    for day_offset in range(min(active_days, 30)):
        day = now - timedelta(days=30 - day_offset)
        n_clicks = random.randint(5, 30)
        for _ in range(n_clicks):
            activity_records.append({
                "user_id": user_id,
                "activity_type": random.choice(["pause_count", "seek_count", "quiz_accuracy", "response_time"]),
                "activity_value": str(round(random.uniform(10, 95), 1)),
                "created_at": (day + timedelta(hours=random.randint(8, 22), minutes=random.randint(0, 59))).isoformat(),
            })
    for i in range(0, len(activity_records), 100):
        supabase.table("student_activity").insert(activity_records[i:i+100]).execute()
    print(f"    student_activity seeded: {len(activity_records)} records")

    # 4. Seed cognitive_metrics
    cog = profile["cognitive"]
    cog_record = {
        "user_id":            user_id,
        "watch_duration":     round(random.uniform(120, 280), 2),
        "pause_count":        random.randint(3, 12),
        "seek_count":         random.randint(1, 8),
        "focus_score":        cog["focus_score"],
        "engagement_score":   cog["engagement_score"],
        "confusion_score":    cog["confusion_score"],
        "learning_velocity":  cog["learning_velocity"],
        "ai_risk_probability": cog["ai_risk_probability"],
        "recommendations": [
            "Cognitive state derived from GRU model (ROC-AUC=0.878).",
            "Data sourced from advanced_persona_profiles.csv pipeline.",
            "Please review full SHAP explanation for feature attribution.",
        ],
    }
    supabase.table("cognitive_metrics").insert(cog_record).execute()
    print(f"    cognitive_metrics seeded: focus={cog['focus_score']}, confusion={cog['confusion_score']}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python seed_eval_profiles.py <user_id> <profile_key>")
        print("profile_key: consistent | last_minute | burnout")
        print("\nAvailable profiles:")
        for k, v in EVAL_PROFILES.items():
            print(f"  {k}: {v['label']}")
        sys.exit(1)

    user_id     = sys.argv[1]
    profile_key = sys.argv[2]

    if profile_key not in EVAL_PROFILES:
        print(f"ERROR: Unknown profile_key '{profile_key}'")
        print(f"Choose from: {list(EVAL_PROFILES.keys())}")
        sys.exit(1)

    ok = seed_eval_profile(user_id, profile_key)
    if ok:
        print(f"\n  Done. Profile '{EVAL_PROFILES[profile_key]['label']}' seeded for user {user_id}")
    else:
        print(f"\n  Failed to seed profile '{profile_key}' for user {user_id}")
        sys.exit(1)
