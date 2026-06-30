from fastapi import APIRouter

from services.persona_service import (
    get_student_persona
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
# GET CURRENT USER PERSONA
# =========================

@router.get("/me/{user_id}")

def get_my_persona(
    user_id: str
):

    return get_student_persona(
        user_id
    )

# =========================
# EXPLAIN PERSONA / RISK
# =========================

@router.get("/explain/{user_id}")

def explain_my_persona(
    user_id: str
):
    from services.explainability_service import get_student_explanation

    return get_student_explanation(
        user_id
    )