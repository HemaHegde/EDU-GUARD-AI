from fastapi import APIRouter
from services.dashboard_service import (
    get_overview_metrics,
    get_engagement_trend,
    get_persona_distribution
)

router = APIRouter()

@router.get("/overview")
def overview():
    """Returns top-level cohort metrics for the dashboard."""
    return get_overview_metrics()

@router.get("/engagement-trend")
def engagement_trend():
    """Returns the 14-day engagement and attention trend line."""
    return get_engagement_trend()

@router.get("/persona-distribution")
def persona_distribution():
    """Returns the breakdown of learner personas in the cohort."""
    return get_persona_distribution()
