
from fastapi import APIRouter

from services.cognitive_service import (

    track_cognitive_behavior,
    get_cognitive_students,
    get_student_cognition,
    cognitive_dashboard

)

router = APIRouter()

# =========================
# HOME
# =========================

@router.get("/")

def cognitive_home():

    return {

        "module":
        "Real Cognitive Intelligence Running"

    }

# =========================
# TRACK LIVE VIDEO BEHAVIOR
# =========================

@router.post("/track")

def track_behavior(
    payload: dict
):

    return track_cognitive_behavior(
        payload
    )

# =========================
# GET ALL STUDENTS
# =========================

@router.get("/students")

def cognitive_students():

    return get_cognitive_students()

# =========================
# GET SINGLE STUDENT
# =========================

@router.get("/student/{user_id}")

def student_cognition(
    user_id: str
):

    return get_student_cognition(
        user_id
    )

# =========================
# DASHBOARD
# =========================

@router.get("/dashboard/overview")

def dashboard():

    return cognitive_dashboard()
