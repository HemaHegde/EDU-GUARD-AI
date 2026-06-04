from config.supabase_client import supabase

import pandas as pd
import os
import joblib

# =========================
# LOAD TRAINED XGBOOST MODEL
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "../../ml/academic_risk_xgboost.pkl"
)

print("MODEL PATH:")
print(MODEL_PATH)

model = joblib.load(MODEL_PATH)

print("XGBoost Model Loaded Successfully")

# =========================
# GET STUDENT RISK SERVICE
# =========================

def get_student_risk_service(user_id: str):

    try:

        # =========================
        # FETCH PROFILE
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
        # FETCH FEATURES
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
                "message": "No engineered features found for student"
            }

        features_db = feature_response.data[0]

        total_clicks = features_db.get(
            "total_clicks",
            0
        )

        avg_score = features_db.get(
            "avg_score",
            0
        )

        active_days = features_db.get(
            "active_days",
            0
        )

        engagement_variability = features_db.get(
            "engagement_variability",
            0
        )

        inactivity_days = features_db.get(
            "inactivity_days",
            0
        )

        engagement_slope = features_db.get(
            "engagement_slope",
            0
        )

        assessment_consistency = features_db.get(
            "assessment_consistency",
            0
        )

        # =========================
        # CREATE MODEL INPUT
        # =========================

        features = pd.DataFrame([
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
                assessment_consistency
            }
        ])

        # =========================
        # XGBOOST PREDICTION
        # =========================

        prediction_probability = float(
            model.predict_proba(
                features
            )[0][1]
        )

        risk_score = int(
            prediction_probability * 100
        )

        # =========================
        # RISK LEVEL
        # =========================

        if risk_score >= 70:

            risk_level = "High"

        elif risk_score >= 40:

            risk_level = "Medium"

        else:

            risk_level = "Low"

        # =========================
        # EXPLAINABLE AI
        # =========================

        reasons = []

        if inactivity_days > 15:

            reasons.append(
                f"High inactivity detected ({inactivity_days} days)"
            )

        if engagement_slope < -0.05:

            reasons.append(
                "Student engagement trend declining"
            )

        if total_clicks < 100:

            reasons.append(
                "Low learning interaction activity"
            )

        if avg_score < 50:

            reasons.append(
                "Academic performance weakening"
            )

        if assessment_consistency > 12:

            reasons.append(
                "Irregular assessment consistency detected"
            )

        if engagement_variability > 8:

            reasons.append(
                "Behavior fluctuation increasing"
            )

        if len(reasons) == 0:

            reasons.append(
                "Student learning behavior stable"
            )

        # =========================
        # PERSONA DETECTION
        # =========================

        if risk_score > 80:

            persona = "Burnout Pattern"

        elif inactivity_days > 20:

            persona = "Passive Watcher"

        elif engagement_slope < -0.08:

            persona = "Anxiety-Spike Learner"

        else:

            persona = "Last-Minute Survivor"

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

            "email":
            student.get(
                "email",
                ""
            ),

            "risk_score":
            risk_score,

            "risk_level":
            risk_level,

            "prediction_probability":
            round(
                prediction_probability,
                4
            ),

            "total_clicks":
            total_clicks,

            "avg_score":
            avg_score,

            "active_days":
            active_days,

            "inactivity_days":
            inactivity_days,

            "engagement_variability":
            engagement_variability,

            "engagement_slope":
            engagement_slope,

            "assessment_consistency":
            assessment_consistency,

            "persona":
            persona,

            "reasons":
            reasons

        }

    except Exception as e:

        return {

            "status":
            "error",

            "message":
            str(e)

        }