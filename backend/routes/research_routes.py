from fastapi import APIRouter
from services.research_service import (
    get_model_metrics,
    get_persona_profiles,
    get_correlations,
    get_research_overview,
    get_shap_importance,
    get_cluster_anova,
)

router = APIRouter()

# =========================
# RESEARCH OVERVIEW
# =========================

@router.get("/overview")
def research_overview():
    """
    Master summary for the research dashboard.
    Returns study title, dataset info, model summary.
    """
    return get_research_overview()

# =========================
# MODEL PERFORMANCE METRICS
# =========================

@router.get("/model-metrics")
def model_metrics():
    """
    Returns XGBoost performance metrics and SHAP feature importance.
    Table 3 equivalent for the research paper.
    """
    return get_model_metrics()

# =========================
# BEHAVIORAL CORRELATIONS
# =========================

@router.get("/correlations")
def correlations():
    """
    Returns Pearson correlation table between behavioral features
    and the disengagement risk outcome.
    Table 2 equivalent for the research paper.
    """
    return get_correlations()

# =========================
# PERSONA PSYCHOLOGICAL PROFILES
# =========================

@router.get("/persona-profiles")
def persona_profiles():
    """
    Returns psychologically grounded profiles for all 6 learner personas,
    including theoretical grounding, constructs, and interventions.
    Table 4 equivalent for the research paper.
    """
    return get_persona_profiles()


# =========================
# SHAP FEATURE IMPORTANCE
# =========================

@router.get("/shap-importance")
def shap_importance():
    """
    Returns SHAP feature importance values for the XGBoost model.
    Figure 1 equivalent for the research paper.
    """
    return get_shap_importance()

# =========================
# CLUSTER ANOVA VALIDATION
# =========================

@router.get("/cluster-anova")
def cluster_anova():
    """
    Returns one-way ANOVA results testing whether K-Means persona
    clusters differ significantly on behavioral features.
    Table 5 equivalent for the research paper.
    """
    return get_cluster_anova()
