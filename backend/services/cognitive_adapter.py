"""
cognitive_adapter.py

Sprint 2 — Cognitive Intelligence integration for the AI Mentor.

PURPOSE
-------
context_builder.py optionally imports `get_cognitive_state(user_id)`.
Until now no module provided that name, so cognitive state was always
None (by design — context_builder never fabricates it).

This file does NOT reimplement any cognitive scoring. All real logic
(GRU model loading, feature-vector construction, focus/engagement/
confusion/learning-velocity scoring, ai_risk_probability, and
recommendation generation) already exists in `cognitive_service.py`
and is reused as-is via its read-only lookup function:

    - latest cognitive record for a user -> cognitive_service.get_student_cognition

`cognitive_service.py` is NOT modified by this sprint. This module is
a thin ADAPTER layer only. Its job is to:
    1. Call the existing, already-computed cognitive record.
    2. Reshape it into exactly the six fields context_builder.py /
       prompt_builder.py need:
           focus_score, engagement_score, confusion_score,
           learning_velocity, ai_risk_probability, recommendations
    3. Return None (never a fabricated value, never a heuristic, never
       an error dict) if no real cognitive record exists for this
       student — so the existing "cognitive_state_available = False"
       fallback path in context_builder.py keeps working exactly as
       it does today.

WHAT THIS FILE DELIBERATELY DOES NOT DO
----------------------------------------
- Does not load or call the GRU model directly.
- Does not compute focus/engagement/confusion/learning-velocity scores
  itself — those are read verbatim from the already-computed record.
- Does not apply any heuristic fallback of its own. If
  cognitive_service.py used its internal heuristic fallback (because
  its GRU model failed to load), that's still a real, already-computed
  record — this adapter just relays it. It does not invent a *second*
  layer of estimation on top.
- Does not raise on failure — context_builder.py already wraps this
  call in try/except, but we fail closed (return None) here too so
  behaviour is correct even if called directly.
"""

from typing import Optional, Dict, Any

from .cognitive_service import get_student_cognition

# The exact set of fields the mentor architecture is allowed to surface.
# Keeps the adapter from leaking unrelated raw DB columns (e.g. internal
# watch_duration/pause_count) into the prompt — only the six scored /
# derived fields the spec asks for.
_EXPOSED_FIELDS = (
    "focus_score",
    "engagement_score",
    "confusion_score",
    "learning_velocity",
    "ai_risk_probability",
    "recommendations",
)


def get_cognitive_state(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Public entry point expected by context_builder.py.

    Returns:
        dict with the six cognitive fields if a real, already-computed
        cognitive record exists for this student, otherwise None.
        Never raises, never fabricates, never estimates.
    """

    try:
        # Reuse existing logic as-is: GRU model invocation (or its own
        # internal heuristic fallback if the model failed to load) and
        # Supabase lookup already happened upstream in
        # track_cognitive_behavior(); here we only read the latest
        # already-persisted result.
        result = get_student_cognition(user_id)
    except Exception:
        # Fail closed — context_builder.py treats None as "genuinely
        # unavailable", which is the correct, honest behaviour here.
        return None

    if not result or not isinstance(result, dict) or "error" in result:
        return None

    # Require that the core scored fields are actually present —
    # if the stored record is malformed/partial, treat as unavailable
    # rather than passing through gaps as fabricated zeros.
    if any(result.get(field) is None for field in _EXPOSED_FIELDS):
        return None

    return {field: result[field] for field in _EXPOSED_FIELDS}
