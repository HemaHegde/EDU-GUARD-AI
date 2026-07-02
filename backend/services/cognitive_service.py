from config.supabase_client import supabase

import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU

# =========================
# GRU COMPATIBILITY SHIM
# =========================
#
# WHY THIS EXISTS:
# The model was originally trained/saved on a TF/Keras
# version where GRU's constructor accepted a 'time_major'
# kwarg. That kwarg was removed in later TF/Keras releases.
# The .h5 file's stored layer config still contains
# 'time_major': False, so a plain load_model() call fails
# with:
#   Unrecognized keyword arguments passed to GRU: {'time_major': False}
#
# HOW IT WORKS:
# Keras reconstructs each layer during loading by calling
# LayerClass(**config). By registering CompatGRU under the
# name "GRU" via custom_objects, this subclass intercepts
# that call, strips the obsolete 'time_major' key, and
# forwards everything else to the real GRU.__init__. The
# resulting layer has the exact same shape/structure as the
# original, so the saved weights load correctly afterward.
# This does NOT change model behavior: time_major=False just
# meant batch-first input ([batch, time, features]), which is
# already what this service feeds the model, and which is the
# only layout current Keras GRU supports anyway.
#
# This is a load-time compatibility fix only. No retraining,
# no architecture changes, no weight changes.
# =========================

class CompatGRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

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

model = None

try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"GRU": CompatGRU}
    )
    print("GRU Cognitive Model Loaded Successfully")
except Exception as model_load_error:
    print(
        f"WARNING: GRU model could not be loaded ({model_load_error}). "
        "Falling back to heuristic scoring."
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
        # DEBUG: Log full raw payload
        # from frontend BEFORE any
        # feature engineering.
        #
        # HOW TO READ THIS:
        # - If values here are 0/null →
        #   frontend stale closure bug.
        #   Fix: ReactPlayer refs (already
        #   done in Cognitive.tsx fix).
        # - If values here are correct →
        #   Supabase insert is the issue.
        #   Check RLS policies or column types.
        # =========================

        print(
            "COGNITIVE PAYLOAD:",
            payload
        )

        print(
            "WATCH:", watch_duration,
            "PAUSE:", pause_count,
            "SEEK:", seek_count
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

        if model is not None:
            prediction = model.predict(
                features,
                verbose=0
            )
            prediction_value = float(
                prediction[0][0]
            )
        else:
            # Heuristic fallback when model unavailable
            prediction_value = float(
                np.clip(
                    (pause_count * 0.05 + seek_count * 0.03) / 10,
                    0.0,
                    1.0
                )
            )

        # =========================
        # SAFE PREDICTION VALUE
        # =========================

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

            # =========================
            # DEBUG: Log exactly what is
            # being inserted into Supabase.
            # If watch_duration/pause_count/
            # seek_count are 0 here but
            # COGNITIVE PAYLOAD showed correct
            # values, there is a type cast
            # or RLS policy blocking the write.
            # =========================

            insert_payload = {

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

            }

            print(
                "SUPABASE INSERT PAYLOAD:",
                insert_payload
            )

            supabase.table(
                "cognitive_metrics"
            ).insert(
                insert_payload
            ).execute()

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
