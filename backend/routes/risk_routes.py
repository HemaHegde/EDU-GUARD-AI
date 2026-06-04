from fastapi import APIRouter

from services.risk_service import (
    get_student_risk_service
)

router = APIRouter()

# =========================
# GET CURRENT STUDENT RISK
# =========================

@router.get("/me/{user_id}")

def get_my_risk(
    user_id: str
):

    return get_student_risk_service(
        user_id
    )