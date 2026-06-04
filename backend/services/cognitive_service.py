from config.supabase_client import supabase

import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import load_model

# =========================
# LOAD TRAINED GRU MODEL
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "../../ml/video_intelligence/gru_cognitive_model.h5"
)

model = load_model(
    MODEL_PATH
)

# =========================
# TRACK LIVE COGNITIVE DATA
# =========================

def track_cognitive_behavior(
    payload: dict
):

    try:

        # =========================
        # EXTRACT PAYLOAD
        # =========================

        user_id = payload.get(
            "user_id"
        )

        watch_duration = float(
            payload.get(
                "watch_duration",
                0
            )
        )

        pause_count = int(
            payload.get(
                "pause_count",
                0
            )
        )

        seek_count = int(
            payload.get(
                "seek_count",
                0
            )
        )

        quiz_accuracy = float(
            payload.get(
                "quiz_accuracy",
                0
            )
        )

        response_time = float(
            payload.get(
                "response_time",
                0
            )
        )

        # =========================
        # CREATE FEATURE VECTOR
        # =========================

        feature_vector = [

            watch_duration,
            pause_count,
            seek_count,
            quiz_accuracy,
            response_time,

            # EXTRA FEATURES
            watch_duration / 10,
            pause_count * 2,
            seek_count * 3

        ]

        # =========================
        # CREATE 100 TIMESTEPS
        # =========================

        sequence = []

        for _ in range(100):

            sequence.append(
                feature_vector
            )

        # =========================
        # FINAL INPUT SHAPE
        # (1,100,8)
        # =========================

        features = np.array(
            [sequence]
        )

        # =========================
        # MODEL PREDICTION
        # =========================

        prediction = model.predict(
            features,
            verbose=0
        )

        # =========================
        # SAFE PREDICTION VALUE
        # =========================

        prediction_value = float(
            prediction[0][0]
        )

        # =========================
        # AI SCORES
        # =========================

        focus_score = int(
            np.clip(
                (1 - prediction_value) * 100,
                0,
                100
            )
        )

        engagement_score = int(
            np.clip(
                (
                    (watch_duration / 300) * 100
                    - (seek_count * 2)
                ),
                0,
                100
            )
        )

        confusion_score = int(
            np.clip(
                (
                    prediction_value * 100
                    + (pause_count * 3)
                ),
                0,
                100
            )
        )

        learning_velocity = int(
            np.clip(
                (
                    quiz_accuracy
                    - response_time
                ),
                0,
                100
            )
        )

        ai_risk_probability = round(
            prediction_value,
            4
        )

        # =========================
        # AI-DRIVEN RECOMMENDATIONS
        # =========================

        recommendations = []

        risk_score = (
            prediction_value * 100
        )

        # HIGH RISK

        if risk_score >= 75:

            recommendations.extend([

                "High cognitive overload detected.",

                "Recommend immediate revision session.",

                "Suggest slowing video playback speed.",

                "Student may require mentor intervention."

            ])

        # MEDIUM RISK

        elif risk_score >= 45:

            recommendations.extend([

                "Moderate confusion patterns detected.",

                "Interactive quizzes may improve retention.",

                "Recommend revisiting difficult concepts."

            ])

        # LOW RISK

        else:

            recommendations.extend([

                "Learning pattern stable.",

                "Student engagement currently healthy.",

                "Continue adaptive learning session."

            ])

        # EXTRA AI INSIGHTS

        if engagement_score < 50:

            recommendations.append(

                "Gamified exercises recommended to improve engagement."

            )

        if focus_score < 50:

            recommendations.append(

                "Frequent pauses indicate reduced concentration."

            )

        if confusion_score > 70:

            recommendations.append(

                "Student showing strong confusion indicators."

            )

        # =========================
        # STORE IN SUPABASE
        # =========================

        try:

            # STEP 3: Added watch_duration, pause_count, seek_count
            supabase.table(
                "cognitive_metrics"
            ).insert({

                "user_id": user_id,

                "watch_duration": watch_duration,

                "pause_count": pause_count,

                "seek_count": seek_count,

                "focus_score": focus_score,

                "engagement_score": engagement_score,

                "confusion_score": confusion_score,

                "learning_velocity": learning_velocity,

                "ai_risk_probability": ai_risk_probability,

                "recommendations": recommendations

            }).execute()

            activity_records = [

                {
                    "user_id": user_id,
                    "activity_type": "pause_count",
                    "activity_value": str(pause_count)
                },

                {
                    "user_id": user_id,
                    "activity_type": "seek_count",
                    "activity_value": str(seek_count)
                },

                {
                    "user_id": user_id,
                    "activity_type": "quiz_accuracy",
                    "activity_value": str(quiz_accuracy)
                },

                {
                    "user_id": user_id,
                    "activity_type": "response_time",
                    "activity_value": str(response_time)
                }

            ]

            supabase.table(
                "student_activity"
            ).insert(
                activity_records
            ).execute()

            # Generate engineered features
            from services.feature_engineering_service import (
                generate_student_features
            )

            generate_student_features(
                user_id
            )

        except Exception as db_error:

            print(
                "Supabase insert error:",
                db_error
            )

        # =========================
        # RESPONSE
        # =========================

        return {

            "status":
            "success",

            "focus_score":
            focus_score,

            "engagement_score":
            engagement_score,

            "confusion_score":
            confusion_score,

            "learning_velocity":
            learning_velocity,

            "ai_risk_probability":
            ai_risk_probability,

            "recommendations":
            recommendations

        }

    except Exception as e:

        return {

            "status":
            "error",

            "message":
            str(e)

        }

# =========================
# GET ALL STUDENTS
# =========================

def get_cognitive_students():

    try:

        response = supabase.table(
            "cognitive_metrics"
        ).select("*").execute()

        return response.data

    except Exception as e:

        return {

            "error":
            str(e)

        }

# =========================
# GET SINGLE STUDENT
# =========================

def get_student_cognition(
    user_id: str
):

    try:

        response = supabase.table(
            "cognitive_metrics"
        ).select("*").eq(
            "user_id",
            user_id
        ).execute()

        if not response.data:

            return {

                "error":
                "Student not found"

            }

        return response.data[-1]

    except Exception as e:

        return {

            "error":
            str(e)

        }

# =========================
# DASHBOARD OVERVIEW
# =========================

def cognitive_dashboard():

    try:

        response = supabase.table(
            "cognitive_metrics"
        ).select("*").execute()

        data = response.data

        if not data:

            return {

                "total_students": 0,
                "average_focus": 0,
                "average_engagement": 0,
                "average_confusion": 0,
                "average_learning_velocity": 0

            }

        df = pd.DataFrame(data)

        # STEP 4: Extended dashboard analytics
        return {

            "total_students":
            len(df),

            "average_focus":
            round(
                df["focus_score"].mean(),
                2
            ),

            "average_engagement":
            round(
                df["engagement_score"].mean(),
                2
            ),

            "average_confusion":
            round(
                df["confusion_score"].mean(),
                2
            ),

            "average_learning_velocity":
            round(
                df["learning_velocity"].mean(),
                2
            ),

            "average_watch_duration":
            round(
                df["watch_duration"].mean(),
                2
            ),

            "average_pause_count":
            round(
                df["pause_count"].mean(),
                2
            ),

            "average_seek_count":
            round(
                df["seek_count"].mean(),
                2
            )

        }

    except Exception as e:

        return {

            "error":
            str(e)

        }