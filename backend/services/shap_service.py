"""
shap_service.py

Sprint 2 — SHAP integration for the AI Mentor.

PURPOSE
-------
context_builder.py optionally imports `get_shap_explanation(user_id)` from
this module. Until now this module didn't exist, so SHAP was always None
(by design — context_builder never fabricates it).

This file does NOT reimplement SHAP computation. All real SHAP logic
(model loading, TreeExplainer, feature lookup from Supabase) already
exists in `explainability_service.py` and is reused as-is:

    - model / explainer loading           -> explainability_service
    - feature vector retrieval for a user -> explainability_service.get_student_explanation
    - SHAP value computation              -> explainability_service.explain_student_psychology

This module is a thin ADAPTER layer only. Its job is to:
    1. Call the existing explainability code.
    2. Reshape its output into the contract context_builder.py /
       prompt_builder.py expect:
           {
               "top_features": [...],
               "feature_importance": {...},
               "summary": "..."
           }
    3. Return None (never a fabricated value, never an error dict) if
       SHAP could not be computed for any reason — so the existing
       "shap_available = False" fallback path in context_builder.py
       keeps working exactly as it does today.

WHAT THIS FILE DELIBERATELY DOES NOT DO
----------------------------------------
- Does not reload or retrain the model.
- Does not recompute SHAP values itself (delegates to TreeExplainer via
  explainability_service).
- Does not invent a "summary" sentence using any number not present in
  the actual SHAP output — the summary is a plain template filled only
  with the top real feature(s) and their real impact direction.
- Does not raise on failure — callers (context_builder.py) already
  wrap this in try/except, but we fail closed (return None) here too
  so behaviour is correct even if called directly.
"""

from typing import Optional, Dict, Any, List

from .explainability_service import get_student_explanation


# How many top features to surface in the mentor-facing payload.
# This only trims the *display* list; nothing about the underlying
# computation changes.
_TOP_N_FEATURES = 5


def _build_summary(top_features: List[Dict[str, Any]]) -> str:
    """
    Builds a short, human-readable summary strictly from the actual
    top SHAP feature(s) already computed upstream. No numbers or
    feature names are invented here — every value comes directly from
    explainability_service's real output.
    """
    if not top_features:
        # Should not normally happen if shap_explanations was non-empty,
        # but fail closed with a neutral statement rather than guessing.
        return "No dominant behavioural driver could be identified."

    leading = top_features[0]
    direction_phrase = (
        "increasing" if leading["impact_direction"] == "positive" else "decreasing"
    )

    summary = (
        f"The strongest factor influencing this student's risk assessment "
        f"is '{leading['feature']}', which is {direction_phrase} their risk."
    )

    if len(top_features) > 1:
        second = top_features[1]
        second_direction = (
            "increasing" if second["impact_direction"] == "positive" else "decreasing"
        )
        summary += (
            f" '{second['feature']}' is also a notable contributor, "
            f"{second_direction} their risk."
        )

    return summary


def get_shap_explanation(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Public entry point expected by context_builder.py.

    Returns:
        dict with keys {top_features, feature_importance, summary} if
        SHAP could genuinely be computed for this student, otherwise
        None. Never raises, never fabricates.
    """

    try:
        # Reuse existing logic: model loading, feature lookup, SHAP
        # computation, and sorting by magnitude are all already done
        # inside explainability_service.py. We do not duplicate any
        # of that here.
        result = get_student_explanation(user_id)
    except Exception:
        # Fail closed — context_builder.py treats None as "genuinely
        # unavailable", which is the correct, honest behaviour here.
        return None

    if not result or result.get("status") != "success":
        return None

    shap_explanations = result.get("shap_explanations") or []
    if not shap_explanations:
        return None

    top_features = shap_explanations[:_TOP_N_FEATURES]

    # feature_importance is a simple {feature_name: shap_value} map,
    # derived directly from the already-computed, already-sorted list.
    # No new computation, just a reshape for prompt_builder.py's
    # generic dict-rendering (see _section_shap_summary).
    feature_importance = {
        item["feature"]: item["shap_value"] for item in top_features
    }

    return {
        "top_features": top_features,
        "feature_importance": feature_importance,
        "summary": _build_summary(top_features),
    }
