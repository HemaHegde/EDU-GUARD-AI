
import pandas as pd
import numpy as np
import os

# =========================
# LOAD DATASET
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(

    BASE_DIR,
    "../../ml/advanced_persona_profiles.csv"

)

persona_df = pd.read_csv(
    DATA_PATH
)

# =========================
# CLEAN DATA
# =========================

persona_df = (
    persona_df
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

# =========================
# GET ALL STUDENTS
# =========================

def get_persona_students():

    return persona_df.head(50).to_dict(
        orient="records"
    )

# =========================
# GET SINGLE PERSONA
# =========================

def get_student_persona(
    student_id: int
):

    student = persona_df[
        persona_df["id_student"] == student_id
    ]

    if student.empty:

        return {

            "error":
            "Student not found"

        }

    student = student.iloc[0]

    return {

        "student_id":
        int(student["id_student"]),

        "persona":
        str(student["persona"]),

        "attention_score":
        float(student["attention_score"]),

        "confusion_score":
        float(student["confusion_score"]),

        "cognitive_overload_score":
        float(
            student[
                "cognitive_overload_score"
            ]
        ),

        "intervention_style":
        str(student["intervention_style"])

    }

# =========================
# DASHBOARD OVERVIEW
# =========================

def persona_dashboard():

    total_students = len(
        persona_df
    )

    persona_distribution = (

        persona_df["persona"]
        .value_counts()
        .to_dict()

    )

    return {

        "total_students":
        total_students,

        "persona_distribution":
        persona_distribution

    }

