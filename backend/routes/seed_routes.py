from fastapi import APIRouter

from services.seed_service import (
    seed_demo_data,
    get_available_profiles
)

router = APIRouter()

# =========================
# GET AVAILABLE PROFILES
# =========================

@router.get("/profiles")
def list_profiles():
    """List available demo student profiles."""
    return get_available_profiles()


# =========================
# SEED DEMO DATA
# =========================

@router.post("/seed/{user_id}")
def seed_data(
    user_id: str,
    profile_type: str = "anxiety_spike"
):
    """
    Seed demo data for a student.

    - user_id: The authenticated user's Supabase UUID
    - profile_type: One of 'anxiety_spike', 'burnout',
                    'consistent', 'passive_watcher'
    """
    return seed_demo_data(
        user_id,
        profile_type
    )
