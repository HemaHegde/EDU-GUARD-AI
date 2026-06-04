
from fastapi import APIRouter

from services.persona_service import (

    get_persona_students,
    get_student_persona,
    persona_dashboard

)

router = APIRouter()

# =========================
# HOME
# =========================

@router.get("/")

def persona_home():

    return {

        "module":
        "Persona Intelligence Running"

    }

# =========================
# GET ALL STUDENTS
# =========================

@router.get("/students")

def persona_students():

    return get_persona_students()

# =========================
# GET SINGLE STUDENT
# =========================

@router.get("/student/{student_id}")

def student_persona(
    student_id: int
):

    return get_student_persona(
        student_id
    )

# =========================
# DASHBOARD
# =========================

@router.get("/dashboard/overview")

def dashboard():

    return persona_dashboard()

