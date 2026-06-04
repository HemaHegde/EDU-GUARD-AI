from config.supabase_client import supabase

import numpy as np
from datetime import datetime


def generate_student_features(user_id: str):

    try:

        response = (
            supabase.table(
                "student_activity"
            )
            .select("*")
            .eq(
                "user_id",
                user_id
            )
            .execute()
        )

        activities = response.data or []

        if len(activities) == 0:

            return None

        # =========================
        # FEATURE VARIABLES
        # =========================

        total_clicks = len(
            activities
        )

        active_days_set = set()

        quiz_scores = []

        daily_activity = {}

        # =========================
        # PROCESS ACTIVITIES
        # =========================

        for activity in activities:

            activity_type = activity.get(
                "activity_type"
            )

            activity_value = activity.get(
                "activity_value"
            )

            created_at = activity.get(
                "created_at"
            )

            if created_at:

                day = created_at[:10]

                active_days_set.add(
                    day
                )

                if day not in daily_activity:

                    daily_activity[day] = 0

                daily_activity[day] += 1

            # =========================
            # QUIZ SCORES
            # =========================

            if activity_type == "quiz_accuracy":

                try:

                    quiz_scores.append(
                        float(
                            activity_value
                        )
                    )

                except:

                    pass

        # =========================
        # ACTIVE DAYS
        # =========================

        active_days = len(
            active_days_set
        )

        inactivity_days = max(
            0,
            30 - active_days
        )

        # =========================
        # SCORES
        # =========================

        if len(quiz_scores) > 0:

            avg_score = round(
                float(
                    np.mean(
                        quiz_scores
                    )
                ),
                2
            )

            assessment_consistency = round(
                float(
                    np.std(
                        quiz_scores
                    )
                ),
                3
            )

        else:

            avg_score = 50.0

            assessment_consistency = 0.0

        # =========================
        # ENGAGEMENT FEATURES
        # =========================

        if len(daily_activity) > 1:

            values = list(
                daily_activity.values()
            )

            engagement_variability = round(
                float(
                    np.std(values)
                ),
                3
            )

            x = np.arange(
                len(values)
            )

            engagement_slope = round(
                float(
                    np.polyfit(
                        x,
                        values,
                        1
                    )[0]
                ),
                4
            )

        else:

            engagement_variability = 0.0

            engagement_slope = 0.0

        # =========================
        # FINAL FEATURE RECORD
        # =========================

        feature_record = {

            "user_id":
            user_id,

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

            "updated_at":
            datetime.utcnow().isoformat()

        }

        # =========================
        # UPSERT FEATURES
        # =========================

        supabase.table(
            "student_features"
        ).upsert(
            feature_record,
            on_conflict="user_id"
        ).execute()

        print(
            "Student Features Updated"
        )

        return feature_record

    except Exception as e:

        print(
            "Feature Engineering Error:",
            e
        )

        return None