import os
import joblib
import pandas as pd

from config.supabase_client import supabase

# =========================
# LOAD PERSONA MODEL
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "../../ml/persona_kmeans.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "../../ml/persona_scaler.pkl"
)

if os.path.exists(MODEL_PATH):
    persona_model = joblib.load(MODEL_PATH)
    print("Persona ML Model Loaded Successfully")
else:
    persona_model = None
    print("Persona Model Not Found")

if os.path.exists(SCALER_PATH):
    persona_scaler = joblib.load(SCALER_PATH)
else:
    persona_scaler = None
    print("Persona Scaler Not Found")

# =========================
# GET STUDENT PERSONA
# =========================

def get_student_persona(user_id: str):

    try:

        # =========================
        # PROFILE
        # =========================

        profile_response = (
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .execute()
        )

        if not profile_response.data:

            return {
                "status": "error",
                "message": "Student profile not found"
            }

        student = profile_response.data[0]

        # =========================
        # STUDENT FEATURES
        # =========================

        feature_response = (
            supabase.table("student_features")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if not feature_response.data:

            return {
                "status": "error",
                "message": "Student features not found"
            }

        features = feature_response.data[0]

        # =========================
        # COGNITIVE METRICS
        # =========================

        cognitive_response = (
            supabase.table("cognitive_metrics")
            .select("*")
            .eq("user_id", user_id)
            .order(
                "created_at",
                desc=True
            )
            .limit(1)
            .execute()
        )

        cognitive = {}

        if cognitive_response.data:

            cognitive = cognitive_response.data[0]

        # =========================
        # FEATURE VALUES
        # =========================

        total_clicks = float(
            features.get(
                "total_clicks",
                0
            ) or 0
        )

        avg_score = float(
            features.get(
                "avg_score",
                0
            ) or 0
        )

        active_days = float(
            features.get(
                "active_days",
                0
            ) or 0
        )

        engagement_variability = float(
            features.get(
                "engagement_variability",
                0
            ) or 0
        )

        inactivity_days = float(
            features.get(
                "inactivity_days",
                0
            ) or 0
        )

        engagement_slope = float(
            features.get(
                "engagement_slope",
                0
            ) or 0
        )

        assessment_consistency = float(
            features.get(
                "assessment_consistency",
                0
            ) or 0
        )

        focus_score = float(
            cognitive.get(
                "focus_score",
                50
            ) or 50
        )

        confusion_score = float(
            cognitive.get(
                "confusion_score",
                50
            ) or 50
        )

        engagement_score = float(
            cognitive.get(
                "engagement_score",
                50
            ) or 50
        )

        ai_risk_probability = float(
            cognitive.get(
                "ai_risk_probability",
                0
            ) or 0
        )

        # Convert probability safely
        if ai_risk_probability <= 1:
            cognitive_overload_score = (
                ai_risk_probability * 100
            )
        else:
            cognitive_overload_score = (
                ai_risk_probability
            )

        # =========================
        # ML FEATURE PREPARATION
        # =========================

        attention_score = focus_score

        boredom_score = max(
            0,
            100 - engagement_score
        )

        # =========================
        # MODEL INPUT
        # =========================

        if persona_model is None or persona_scaler is None:

            return {
                "status": "error",
                "message": "Persona model not loaded"
            }

        model_input = pd.DataFrame([
            {
                "total_clicks":
                total_clicks,

                "avg_score":
                avg_score,

                "active_days":
                active_days,

                "engagement_variability":
                engagement_variability,

                "inactivity_days":
                inactivity_days,

                "engagement_slope":
                engagement_slope,

                "assessment_consistency":
                assessment_consistency,

                "attention_score":
                attention_score,

                "confusion_score":
                confusion_score,

                "boredom_score":
                boredom_score,

                "cognitive_overload_score":
                cognitive_overload_score
            }
        ])

        # =========================
        # SCALE FEATURES
        # =========================

        scaled_features = (
            persona_scaler.transform(
                model_input
            )
        )

        # =========================
        # PREDICT CLUSTER
        # =========================

        cluster = int(
            persona_model.predict(
                scaled_features
            )[0]
        )

        # =========================
        # CLUSTER → PERSONA
        # =========================

        persona_map = {

            0: "Silent Isolator",

            1: "Burnout Pattern",

            2: "Anxiety-Spike Learner",

            3: "Passive Watcher",

            4: "Consistent Learner",

            5: "Last-Minute Survivor"

        }

        persona = persona_map.get(
            cluster,
            "Unknown Persona"
        )

        # =========================
        # INTERVENTION
        # =========================

        intervention_map = {

            "Silent Isolator":
            "Social engagement support",

            "Burnout Pattern":
            "Mental wellness intervention",

            "Anxiety-Spike Learner":
            "Guided pacing and stress reduction",

            "Passive Watcher":
            "Interactive participation encouragement",

            "Consistent Learner":
            "Advanced learning opportunities",

            "Last-Minute Survivor":
            "Deadline management coaching"

        }

        intervention = (
            intervention_map.get(
                persona,
                "General academic support"
            )
        )

        # =========================
        # RESPONSE
        # =========================

        return {

            "status":
            "success",

            "student_name":
            student.get(
                "full_name",
                "Unknown Student"
            ),

            "persona":
            persona,

            "attention_score":
            round(
                attention_score,
                2
            ),

            "confusion_score":
            round(
                confusion_score,
                2
            ),

            "cognitive_overload_score":
            round(
                cognitive_overload_score,
                2
            ),

            "engagement_score":
            round(
                engagement_score,
                2
            ),

            "avg_score":
            avg_score,

            "active_days":
            active_days,

            "total_clicks":
            total_clicks,

            "cluster":
            cluster,

            "intervention_style":
            intervention

        }

    except Exception as e:

        return {

            "status":
            "error",

            "message":
            str(e)

        }
