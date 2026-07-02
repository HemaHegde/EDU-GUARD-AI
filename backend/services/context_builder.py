"""
context_builder.py

Responsible for ONE thing: gathering all the raw context needed to talk to
a student, from whatever services already exist in the backend, and
packaging it into a single, predictable StudentContext object.

Design rules followed here (per Sprint 1 spec):
- Never invent SHAP values or cognitive-state values. If the upstream
  service that would provide them is unavailable, missing, or errors out,
  the corresponding field stays None.
- Does not talk to the LLM. Does not build prompts. Does not touch FAISS.
  Single responsibility = data gathering.
- Safe to call from the existing mentor_service without changing its
  public signature.

SPRINT 2 UPDATE:
- Cognitive state is now wired to the real GRU-based cognitive
  intelligence pipeline via `services/cognitive_adapter.py`, which
  thinly wraps the existing, unmodified `cognitive_service.py`
  (`get_student_cognition`). No new scoring, no heuristics, no
  fabricated values were added here — only the import target changed
  from the placeholder `cognitive_state_service` to the real adapter.
- SHAP wiring (added in Sprint 2 SHAP work) is unchanged.

TIMING INSTRUMENTATION (added — no behavioural changes):
- Wrapped the profile lookup, persona/risk/SHAP/cognitive calls, and
  chat-history lookup in high-resolution `time.perf_counter()` timers.
- Per-stage durations are stored in `StudentContext._timings` (a plain
  internal dict, not part of any public API contract). This is purely
  additive: no query, no control flow, no return value, and no field
  that was previously present has been altered. `mentor_service.py`
  reads `context._timings` to fold these numbers into its own timing
  summary; nothing else consumes this field, and it is never sent to
  the LLM, persisted, or included in the FastAPI response payload.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from config.supabase_client import supabase

from services.risk_service import get_student_risk_service
from services.persona_service import get_student_persona

# =========================================================
# OPTIONAL INTEGRATIONS
# =========================================================
# These two are genuinely optional. If your project does not yet have
# a SHAP explanation service or a cognitive-state service, these imports
# will fail and we fall back to None — we do NOT fabricate replacement
# values. Wire up the real services here when they exist.

try:
    from services.shap_service import get_shap_explanation
    _SHAP_AVAILABLE = True
except Exception:
    get_shap_explanation = None
    _SHAP_AVAILABLE = False

try:
    # Sprint 2: real cognitive intelligence, via a lightweight adapter
    # over the existing (unmodified) cognitive_service.py. The adapter
    # never computes scores itself — it only relays an already-computed
    # record or returns None if none exists.
    from services.cognitive_adapter import get_cognitive_state
    _COGNITIVE_STATE_AVAILABLE = True
except Exception:
    get_cognitive_state = None
    _COGNITIVE_STATE_AVAILABLE = False


# =========================================================
# DATA CONTRACT
# =========================================================

@dataclass
class StudentContext:
    """
    Single, predictable container for everything the mentor might need
    to know about a student before generating a response.

    Any field that could not be sourced from a real upstream service is
    left as None (or an empty list/dict) rather than guessed at.
    """

    user_id: str

    # Profile
    student_name: str = "Student"

    # Persona
    persona: str = "Unknown Persona"
    intervention_style: str = ""

    # Academic risk
    risk_score: Optional[float] = None
    risk_level: str = "Unknown"
    risk_reasons: List[str] = field(default_factory=list)

    # SHAP explanation (optional — None if unavailable)
    shap_explanation: Optional[Dict[str, Any]] = None
    shap_available: bool = False

    # Cognitive state (optional — None if unavailable)
    # Sprint 2: when available, this dict has exactly the six fields
    # produced by the GRU cognitive pipeline (via cognitive_adapter.py):
    #   focus_score, engagement_score, confusion_score,
    #   learning_velocity, ai_risk_probability, recommendations
    cognitive_state: Optional[Dict[str, Any]] = None
    cognitive_state_available: bool = False

    # Behaviour summary (derived from risk_reasons / persona, not invented
    # numbers — see _build_behaviour_summary)
    behaviour_summary: str = ""

    # Conversation history (already formatted, oldest -> newest)
    chat_history: str = ""

    # Retrieved learning material context (filled in by the retrieval
    # step, kept here so the whole context lives in one object)
    retrieved_chunks: List[Dict[str, str]] = field(default_factory=list)

    # -------------------------
    # TIMING INSTRUMENTATION ONLY
    # -------------------------
    # Internal-only bag of per-stage durations (seconds) collected while
    # building this context. Not part of the data contract consumed by
    # prompt_builder.py or confidence.py, not persisted, not returned
    # to API callers. mentor_service.py reads this purely for logging.
    _timings: Dict[str, float] = field(default_factory=dict)


# =========================================================
# BUILDER
# =========================================================

def build_student_context(
    user_id: str,
    history_limit: int = 5,
) -> StudentContext:
    """
    Gathers student profile, risk, persona, SHAP (if available),
    cognitive state (if available), and recent chat history.

    Never raises for missing optional data — only raises if a REQUIRED
    upstream call (profile lookup) fails outright, since the original
    mentor_service treated that as fatal too (wrapped in its own
    try/except at the call site).

    TIMING: each stage below is wrapped with time.perf_counter() and the
    duration recorded into context._timings. This is purely additive
    instrumentation — the order of operations, the calls made, and the
    values assigned to context fields are all unchanged from before.
    """

    context = StudentContext(user_id=user_id)

    # -------------------------
    # Student profile
    # -------------------------
    _t0 = time.perf_counter()

    profile_response = (
        supabase.table("profiles")
        .select("*")
        .eq("id", user_id)
        .execute()
    )

    if profile_response.data:
        context.student_name = (
            profile_response.data[0].get("full_name", "Student")
        )

    context._timings["profile"] = time.perf_counter() - _t0

    # -------------------------
    # Persona
    # -------------------------
    _t0 = time.perf_counter()

    persona_data = get_student_persona(user_id) or {}
    context.persona = persona_data.get("persona", "Unknown Persona")
    context.intervention_style = persona_data.get("intervention_style", "")

    context._timings["persona"] = time.perf_counter() - _t0

    # -------------------------
    # Academic risk
    # -------------------------
    _t0 = time.perf_counter()

    risk_data = get_student_risk_service(user_id) or {}
    context.risk_score = risk_data.get("risk_score")
    context.risk_level = risk_data.get("risk_level", "Unknown")
    context.risk_reasons = risk_data.get("reasons", []) or []

    context._timings["risk"] = time.perf_counter() - _t0

    # -------------------------
    # SHAP explanation (optional, never fabricated)
    # -------------------------
    _t0 = time.perf_counter()

    if _SHAP_AVAILABLE and get_shap_explanation is not None:
        try:
            shap_result = get_shap_explanation(user_id)
            if shap_result:
                context.shap_explanation = shap_result
                context.shap_available = True
        except Exception:
            # Any failure here means we genuinely don't have SHAP data.
            # Leave as None — do not guess.
            context.shap_explanation = None
            context.shap_available = False

    context._timings["shap"] = time.perf_counter() - _t0

    # -------------------------
    # Cognitive state (optional, never fabricated)
    # -------------------------
    _t0 = time.perf_counter()

    if _COGNITIVE_STATE_AVAILABLE and get_cognitive_state is not None:
        try:
            cog_result = get_cognitive_state(user_id)
            if cog_result:
                context.cognitive_state = cog_result
                context.cognitive_state_available = True
        except Exception:
            context.cognitive_state = None
            context.cognitive_state_available = False

    context._timings["cognitive"] = time.perf_counter() - _t0

    # -------------------------
    # Behaviour summary
    # -------------------------
    # Built only from data we actually have (risk_reasons, persona).
    # This is a textual rollup, not a new numeric feature — nothing here
    # is invented, it's a restatement of risk_service's own output.
    # (Negligible cost — not separately timed, matches original scope.)
    context.behaviour_summary = _build_behaviour_summary(
        persona=context.persona,
        risk_level=context.risk_level,
        risk_reasons=context.risk_reasons,
    )

    # -------------------------
    # Chat history
    # -------------------------
    _t0 = time.perf_counter()

    history_response = (
        supabase.table("mentor_history")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(history_limit)
        .execute()
    )

    if history_response.data:
        formatted = []
        for row in reversed(history_response.data):
            formatted.append(
                f"Student: {row['question']}\nAura: {row['response']}\n"
            )
        context.chat_history = "\n".join(formatted)

    context._timings["history"] = time.perf_counter() - _t0

    return context


def _build_behaviour_summary(
    persona: str,
    risk_level: str,
    risk_reasons: List[str],
) -> str:
    """
    Produces a short, human-readable behaviour summary strictly from
    fields we already have evidence for. If there are no risk_reasons,
    says so plainly instead of inventing a narrative.
    """

    if not risk_reasons:
        return (
            f"No specific behavioural risk factors are currently on "
            f"record for this student (persona: {persona}, "
            f"risk level: {risk_level})."
        )

    reasons_text = "; ".join(risk_reasons)
    return (
        f"Persona: {persona}. Risk level: {risk_level}. "
        f"Contributing factors on record: {reasons_text}."
    )
