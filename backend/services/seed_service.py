"""
Demo Seed Service
=================
Populates realistic sample data for a logged-in user so that
Dashboard, Risk, Persona, and Cognitive pages work during a
live demo without requiring the full video-watching pipeline.

The data represents a psychologically realistic "Anxiety-Spike Learner"
student profile — ideal for demonstrating the system's detection,
explainability, and intervention capabilities.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from config.supabase_client import supabase


# ═══════════════════════════════════════════════════════════════
#  DEMO STUDENT PROFILES
#  Each profile represents a different persona archetype
#  for a richer demonstration
# ═══════════════════════════════════════════════════════════════

DEMO_PROFILES = {
    "anxiety_spike": {
        "label": "Anxiety-Spike Learner",
        "features": {
            "total_clicks": 312,
            "avg_score": 62.5,
            "active_days": 18,
            "engagement_variability": 11.42,
            "inactivity_days": 12,
            "engagement_slope": -0.12,
            "assessment_consistency": 14.7,
        },
        "cognitive": {
            "focus_score": 58,
            "engagement_score": 52,
            "confusion_score": 61,
            "learning_velocity": 44,
            "ai_risk_probability": 0.67,
        },
    },
    "burnout": {
        "label": "Burnout Pattern",
        "features": {
            "total_clicks": 87,
            "avg_score": 41.2,
            "active_days": 8,
            "engagement_variability": 15.3,
            "inactivity_days": 22,
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
    "passive_watcher": {
        "label": "Passive Watcher",
        "features": {
            "total_clicks": 45,
            "avg_score": 38.0,
            "active_days": 5,
            "engagement_variability": 3.1,
            "inactivity_days": 25,
            "engagement_slope": -0.02,
            "assessment_consistency": 7.5,
        },
        "cognitive": {
            "focus_score": 41,
            "engagement_score": 22,
            "confusion_score": 55,
            "learning_velocity": 30,
            "ai_risk_probability": 0.78,
        },
    },
}


def seed_demo_data(user_id: str, profile_type: str = "anxiety_spike"):
    """
    Seeds realistic demo data for a student.

    Parameters
    ----------
    user_id : str
        The Supabase auth user ID to seed data for.
    profile_type : str
        One of: 'anxiety_spike', 'burnout', 'consistent', 'passive_watcher'

    Returns
    -------
    dict with status and summary of seeded data.
    """

    try:

        profile = DEMO_PROFILES.get(profile_type)
        if not profile:
            return {
                "status": "error",
                "message": f"Unknown profile type: {profile_type}. "
                           f"Choose from: {list(DEMO_PROFILES.keys())}"
            }

        # ─── 1. Verify user exists in profiles ────────────────────

        profile_response = (
            supabase.table("profiles")
            .select("id, full_name")
            .eq("id", user_id)
            .execute()
        )

        if not profile_response.data:
            return {
                "status": "error",
                "message": "User not found in profiles table. "
                           "Please sign up first via the frontend."
            }

        student_name = profile_response.data[0].get(
            "full_name", "Demo Student"
        )

        # ─── 2. Seed student_features ─────────────────────────────

        feature_record = {
            "user_id": user_id,
            **profile["features"],
            "updated_at": datetime.utcnow().isoformat(),
        }

        supabase.table("student_features").upsert(
            feature_record,
            on_conflict="user_id"
        ).execute()

        # ─── 3. Seed student_activity (realistic timeline) ────────

        now = datetime.utcnow()
        activity_records = []
        active_days = profile["features"]["active_days"]

        for day_offset in range(active_days):

            day = now - timedelta(days=30 - day_offset)

            # Simulate varied daily activity
            n_clicks = random.randint(3, 22)

            for _ in range(n_clicks):
                activity_records.append({
                    "user_id": user_id,
                    "activity_type": random.choice([
                        "pause_count",
                        "seek_count",
                        "quiz_accuracy",
                        "response_time",
                    ]),
                    "activity_value": str(
                        round(random.uniform(10, 95), 1)
                    ),
                    "created_at": (
                        day + timedelta(
                            hours=random.randint(8, 22),
                            minutes=random.randint(0, 59)
                        )
                    ).isoformat(),
                })

        # Insert in batches (Supabase limit)
        batch_size = 100
        for i in range(0, len(activity_records), batch_size):
            batch = activity_records[i:i + batch_size]
            supabase.table("student_activity").insert(
                batch
            ).execute()

        # ─── 4. Seed cognitive_metrics ────────────────────────────

        cognitive_record = {
            "user_id": user_id,
            "watch_duration": random.uniform(120, 280),
            "pause_count": random.randint(3, 12),
            "seek_count": random.randint(1, 8),
            **profile["cognitive"],
            "recommendations": [
                "Moderate confusion patterns detected.",
                "Interactive quizzes may improve retention.",
                "Recommend revisiting difficult concepts."
            ],
        }

        supabase.table("cognitive_metrics").insert(
            cognitive_record
        ).execute()

        # ─── 5. Done ─────────────────────────────────────────────

        return {
            "status": "success",
            "message": f"Demo data seeded for {student_name}",
            "profile_type": profile_type,
            "persona_label": profile["label"],
            "seeded_tables": [
                "student_features",
                "student_activity",
                "cognitive_metrics",
            ],
            "activity_records_created": len(activity_records),
            "features": profile["features"],
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


def get_available_profiles():
    """Returns available demo profiles for the seed selector."""
    return {
        profile_id: {
            "label": data["label"],
            "description": f"Seeds data representing a {data['label']} student",
            "risk_level": "High" if data["cognitive"]["ai_risk_probability"] >= 0.65
                          else "Medium" if data["cognitive"]["ai_risk_probability"] >= 0.35
                          else "Low",
        }
        for profile_id, data in DEMO_PROFILES.items()
    }
