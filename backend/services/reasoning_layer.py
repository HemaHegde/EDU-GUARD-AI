"""
reasoning_layer.py

Sprint 7 — Multi-signal reasoning architecture for the AI Mentor.

SCOPE / HARD CONSTRAINTS FOR THIS SPRINT
-----------------------------------------
- No Supabase schema changes: no new tables, no new columns, no new
  persistent store of any kind. `mentor_history` is unchanged.
- No API route changes. `ask_mentor(user_id, question)` keeps its
  existing signature; every existing key in its return dict is
  preserved exactly. Only new, additive keys are introduced.
- Persona Memory and Recommendation Diversity are derived ENTIRELY
  from `context.chat_history` (already gathered by context_builder.py
  from the existing `mentor_history` table via a normal SELECT). No
  new table, no new query, no write beyond the existing history
  insert already performed by mentor_service.py.
- Nothing here fabricates a new fact about the student. Every function
  below either (a) re-ranks/re-weights data StudentContext already
  holds, or (b) pattern-matches against the mentor's own PAST replies
  (which are real, already-stored text) to avoid repeating itself.
  Where something genuinely cannot be derived honestly from available
  data (see `previous_improvement` note below), it is left out rather
  than guessed at — same "never fabricate" principle the rest of this
  codebase follows (context_builder.py, confidence.py).

WHAT'S IMPLEMENTED HERE
------------------------
1. Evidence Fusion       -> build_evidence_profile()
2. Conversation Planning -> build_conversation_plan()
3. Evidence Confidence
   Weighting              -> compute_evidence_confidence()
4. Persona Memory +
   Recommendation
   Diversity (derived
   from chat_history)     -> derive_persona_memory()
5. Persona-varied
   few-shot bank          -> get_few_shot_example()
6. Retrieval Safety
   Classification
   (Sprint 8 addition)    -> classify_retrieval_safety()

SPRINT 8 UPDATE (HALLUCINATION SAFETY HARDENING):
Adds item 6 above, `classify_retrieval_safety()`, a pure function that
turns retrieval-quality numbers already computed by retrieval.py
(similarity) and confidence.py (normalized confidence, contradiction
detection) into one of SAFE / UNCERTAIN / NEEDS_REVIEW for API
consumers and educator-review tooling. It takes only primitives as
arguments (no StudentContext, no cross-module imports), preserving
this module's existing "dependency-light function library" design.
Nothing else in this file was changed by Sprint 8.

Item 5 ("Response Templates per Persona") from the improvement brief
is intentionally NOT a separate template dict here — it already exists
as `_PSYCHOLOGICAL_FRAMEWORKS` / `_PERSONA_ACTIONS` in prompt_builder.py
(Sprint 1 / Sprint 5). This module reuses those (passed in as
arguments, not re-imported, to avoid a circular import with
prompt_builder.py) inside `build_conversation_plan()` rather than
duplicating a second persona table.

WHY THIS FILE DOES NOT IMPORT prompt_builder.py
--------------------------------------------------
prompt_builder.py imports THIS module to render its new sections. If
this module also imported prompt_builder.py, that would be a circular
import. Instead, anything this module needs that already lives in
prompt_builder.py (persona actions, framework text, the priority-focus
result) is passed in as a plain argument by the caller
(prompt_builder.py), which already has it. This module stays a pure,
dependency-light function library: it takes StudentContext + already-
computed values in, and returns plain dicts/strings out.

ON "previous_improvement"
--------------------------
The improvement brief's Persona Memory sketch lists "previous
improvement" as a field. There is no honest way to derive that from
`context.chat_history` alone — that text contains what the mentor SAID,
not a trend in the student's actual scores over time (which would
require reading historical risk/cognitive records, i.e. new queries
against tables this sprint is not touching). Rather than fabricate a
plausible-sounding "you're improving!" claim with no evidence behind
it, this module deliberately omits that field. `derive_persona_memory()`
only returns what can be truthfully read off the stored conversation:
the previous opening phrasing, the previous student-stated struggle,
and which known recommendation "tags" were already used.
"""

import re
from typing import Any, Dict, List, Optional

from .context_builder import StudentContext


# =========================================================
# 1. EVIDENCE FUSION
# =========================================================
# Turns the single PRIMARY signal already chosen by
# prompt_builder.determine_priority_focus() into a ranked profile:
# PRIMARY / SECONDARY / SUPPORTING / lower-priority-this-turn.
#
# "Strength" below is a RENDERING/RANKING heuristic only — it decides
# reading order for the LLM, nothing else. It is derived strictly from
# fields StudentContext already holds (risk_score, SHAP magnitudes,
# cognitive scores, persona, retrieved-chunk count); no new upstream
# call, no new stored value, no new fact about the student.

# Maps a determine_priority_focus() `id` onto the evidence-candidate
# `id` below, so the signal already chosen as PRIMARY is not also
# double-counted as SECONDARY/SUPPORTING.
_FOCUS_ID_TO_CANDIDATE_ID = {
    "confusion": "confusion",
    "burnout": "persona",
    "engagement": "engagement",
    "inactivity": "shap",
    "enrichment": "persona",
    "persona_default": "persona",
}

_FOCUS_ID_TO_LABEL = {
    "confusion": "Confusion (cognitive state)",
    "burnout": "Burnout Pattern + High Risk",
    "engagement": "Low Engagement (cognitive state)",
    "inactivity": "SHAP: Inactivity as top risk driver",
    "enrichment": "Consistent Learner / Enrichment",
    "persona_default": None,  # filled in with the actual persona name at call time
}


def _focus_label(context: StudentContext, focus: Dict[str, str]) -> str:
    label = _FOCUS_ID_TO_LABEL.get(focus["id"])
    if label is not None:
        return label
    return f"Persona: {context.persona}"


def _evidence_candidates(context: StudentContext) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if context.risk_score is not None:
        candidates.append({
            "id": "risk",
            "label": f"Academic Risk ({context.risk_level}, score {context.risk_score})",
            "strength": float(context.risk_score),
        })

    if context.shap_available and context.shap_explanation:
        top_features = (context.shap_explanation or {}).get("top_features") or []
        if top_features:
            top = top_features[0]
            magnitudes = [abs(f.get("impact_magnitude") or 0) for f in top_features]
            total_mag = sum(magnitudes) or 1.0
            relative_share = abs(top.get("impact_magnitude") or 0) / total_mag
            candidates.append({
                "id": "shap",
                "label": f"SHAP driver: {top.get('feature', 'unknown')}",
                "strength": relative_share * 100.0,
            })

    cognitive = context.cognitive_state if context.cognitive_state_available else None
    if cognitive and cognitive.get("confusion_score") is not None:
        candidates.append({
            "id": "confusion",
            "label": "Confusion (cognitive state)",
            "strength": float(cognitive["confusion_score"]),
        })
    if cognitive and cognitive.get("engagement_score") is not None:
        # Low engagement is the concerning direction, so strength rises
        # as engagement_score falls.
        candidates.append({
            "id": "engagement",
            "label": "Low engagement (cognitive state)",
            "strength": max(0.0, 100.0 - float(cognitive["engagement_score"])),
        })

    if context.persona and context.persona != "Unknown Persona":
        persona_strength = 45.0
        if context.persona == "Burnout Pattern" and context.risk_level == "High":
            persona_strength = 90.0
        candidates.append({
            "id": "persona",
            "label": f"Persona: {context.persona}",
            "strength": persona_strength,
        })

    n_chunks = len(context.retrieved_chunks or [])
    if n_chunks:
        candidates.append({
            "id": "material",
            "label": "Retrieved learning material",
            "strength": min(n_chunks * 15.0, 60.0),
        })

    return candidates


def build_evidence_profile(context: StudentContext, focus: Dict[str, str]) -> Dict[str, Any]:
    """
    Returns:
        {
            "primary":    {"label": str, "reason": str},
            "secondary":  {"id", "label", "strength"} | None,
            "supporting": {"id", "label", "strength"} | None,
            "ignored":    [ {"id", "label", "strength"}, ... ],
        }

    `focus` must be the output of prompt_builder.determine_priority_focus(context)
    — passed in rather than recomputed here, so PRIMARY always agrees
    exactly with the existing PRIORITY FOCUS section and with
    mentor_service.py's fallback-recommendation logic (both already
    call determine_priority_focus and must stay consistent with it).
    """
    candidates = _evidence_candidates(context)
    primary_candidate_id = _FOCUS_ID_TO_CANDIDATE_ID.get(focus["id"])

    remaining = [c for c in candidates if c["id"] != primary_candidate_id]
    remaining.sort(key=lambda c: c["strength"], reverse=True)

    return {
        "primary": {"label": _focus_label(context, focus), "reason": focus["reason"]},
        "secondary": remaining[0] if len(remaining) > 0 else None,
        "supporting": remaining[1] if len(remaining) > 1 else None,
        "ignored": remaining[2:],
    }


# =========================================================
# 2. CONVERSATION STRATEGY PLANNER
# =========================================================
# Translates the same PRIMARY focus into a short, concrete plan for
# HOW to respond (not just WHAT to prioritize), so the LLM follows a
# plan instead of inventing its own structure each turn.

_FOCUS_TO_PLAN = {
    "confusion": {
        "opening_goal": "Reassure briefly, then clarify",
        "teaching_style": "Explain slowly, one step at a time; confirm understanding before adding anything new",
        "recommendation_style": "One small clarifying action only",
        "educator_goal": "Flag the concept for re-teaching or a worked example",
    },
    "burnout": {
        "opening_goal": "Reassure and validate; do not push more work",
        "teaching_style": "Brief, low-pressure, permission-giving tone",
        "recommendation_style": "One recovery-oriented micro-action",
        "educator_goal": "Flag for a workload/wellness check-in",
    },
    "engagement": {
        "opening_goal": "Invite active participation",
        "teaching_style": "Socratic; offer a choice rather than a mandate",
        "recommendation_style": "One small interactive action, not passive review",
        "educator_goal": "Prompt with an interactive task rather than more reading",
    },
    "inactivity": {
        "opening_goal": "Welcome back, non-judgmental",
        "teaching_style": "Focus on restarting momentum, not catching up all at once",
        "recommendation_style": "One tiny piece of coursework today, to break the streak",
        "educator_goal": "Reach out directly given the inactivity pattern",
    },
    "enrichment": {
        "opening_goal": "Affirm what's already working",
        "teaching_style": "Forward-looking and growth-oriented; no risk or remedial framing",
        "recommendation_style": "One stretch/enrichment opportunity",
        "educator_goal": "Consider for a peer-mentoring or leadership role",
    },
}


def build_conversation_plan(
    context: StudentContext,
    focus: Dict[str, str],
    persona_actions: Dict[str, str],
    framework_text: str,
) -> Dict[str, str]:
    """
    `persona_actions` and `framework_text` are passed in by
    prompt_builder.py (it already computes both via
    get_persona_actions() / _get_psychological_framework()) so this
    module never needs to import prompt_builder.py itself.
    """
    plan = _FOCUS_TO_PLAN.get(focus["id"])
    if plan is not None:
        return plan

    # focus["id"] == "persona_default": no single signal dominated, so
    # the plan is built directly from the persona's own strategy —
    # same source of truth the PRIORITY FOCUS / Persona sections
    # already use, just reshaped into the four planning fields.
    return {
        "opening_goal": "Lead with the learner's own behavioural pattern",
        "teaching_style": framework_text,
        "recommendation_style": "One action grounded in the persona strategy above",
        "educator_goal": persona_actions.get(
            "educator", "Monitor and re-check after more activity data is available"
        ),
    }


# =========================================================
# 3. EVIDENCE CONFIDENCE WEIGHTING
# =========================================================
# Produces a 0-1 confidence weight per evidence source, so the prompt
# can tell the LLM not just WHAT the evidence says but HOW MUCH to
# lean on it. These are heuristic weights about evidence QUALITY
# (distance from a decision boundary, relative SHAP share, plain
# presence/absence) — not new facts about the student, and not a
# replacement for confidence.py's High/Medium/Low response-level
# rubric, which is unchanged and still the one persisted/returned by
# the API.

def compute_evidence_confidence(context: StudentContext) -> Dict[str, float]:
    weights: Dict[str, float] = {}

    if context.risk_score is not None:
        # Further from the 50/100 midpoint = less ambiguous prediction.
        weights["risk"] = round(min(abs(context.risk_score - 50) / 50.0, 1.0), 2)
    else:
        weights["risk"] = 0.0

    weights["persona"] = 1.0 if context.persona not in (None, "", "Unknown Persona") else 0.0

    if context.shap_available and context.shap_explanation:
        top_features = (context.shap_explanation or {}).get("top_features") or []
        if top_features:
            magnitudes = [abs(f.get("impact_magnitude") or 0) for f in top_features]
            total = sum(magnitudes) or 1.0
            weights["shap"] = round(abs(top_features[0].get("impact_magnitude") or 0) / total, 2)
        else:
            weights["shap"] = 0.0
    else:
        weights["shap"] = 0.0

    weights["cognitive"] = 1.0 if (context.cognitive_state_available and context.cognitive_state) else 0.0

    return weights


# =========================================================
# 4. PERSONA MEMORY + RECOMMENDATION DIVERSITY
# =========================================================
# Derived ENTIRELY from context.chat_history — the already-fetched,
# already-formatted text context_builder.py builds from the existing
# mentor_history table (see context_builder.py's "Chat history"
# stage). No new table, column, or query.
#
# context.chat_history's format (fixed by context_builder.py, not
# touched here) is a sequence of:
#     "Student: <question>\nAura: <response>\n"
# turns joined by "\n". The pattern below relies only on that existing,
# unchanged convention.

_TURN_PATTERN = re.compile(
    r"Student:\s*(.*?)\nAura:\s*(.*?)(?=\nStudent:|\Z)", re.DOTALL
)

# Known recommendation "tags" -> substrings that would appear in the
# mentor's own past replies if that recommendation was given before.
# Sourced directly from prompt_builder.py's own _PERSONA_ACTIONS /
# determine_priority_focus() action text (see prompt_builder.py) — not
# a new vocabulary, just a lookup table for recognizing text the
# system itself already generates.
_KNOWN_ACTION_TAGS = {
    "recovery_break": ("recovery break", "reduce your workload"),
    "interactive_task": ("discussion post", "short quiz", "interactive action", "quiz question"),
    "shorten_session": ("shorten your next study session", "short break partway"),
    "forum_question": ("course forum", "ask one question", "classmate"),
    "if_then_plan": ("if-then plan", "specific plan", "space out revision"),
    "enrichment": ("enrichment activity", "advanced topic", "peer-mentoring", "leadership"),
    "clarify_concept": ("specific step", "concept that's unclear", "worked example"),
    "resume_activity": ("log in and complete", "inactivity streak"),
}


def _extract_turns(chat_history: str) -> List[Dict[str, str]]:
    if not chat_history:
        return []
    matches = _TURN_PATTERN.findall(chat_history)
    return [{"student": q.strip(), "aura": a.strip()} for q, a in matches]


def derive_persona_memory(chat_history: str) -> Dict[str, Any]:
    """
    Returns:
        {
            "has_history": bool,
            "previous_opening": str | None,   # first ~6 words of the mentor's last reply
            "previous_struggle": str | None,  # the student's own last question, truncated
            "used_action_tags": [str, ...],   # recommendation tags already seen in history
        }

    Everything here is read directly off real, already-stored text.
    Nothing is inferred about the student's internal state or scored;
    see the module docstring for why "previous_improvement" is
    deliberately not included.
    """
    turns = _extract_turns(chat_history)
    if not turns:
        return {
            "has_history": False,
            "previous_opening": None,
            "previous_struggle": None,
            "used_action_tags": [],
        }

    last_turn = turns[-1]
    previous_opening = (
        " ".join(last_turn["aura"].split()[:6]) if last_turn["aura"] else None
    )

    used_tags = set()
    for turn in turns:
        text_lower = turn["aura"].lower()
        for tag, phrases in _KNOWN_ACTION_TAGS.items():
            if any(phrase in text_lower for phrase in phrases):
                used_tags.add(tag)

    return {
        "has_history": True,
        "previous_opening": previous_opening,
        "previous_struggle": last_turn["student"][:160] if last_turn["student"] else None,
        "used_action_tags": sorted(used_tags),
    }


# =========================================================
# 5. PERSONA-VARIED FEW-SHOT BANK
# =========================================================
# One tiny (3-4 line) STRUCTURAL example per persona, still using
# bracketed placeholders (never naturalistic filled-in prose) so the
# Sprint 6 fix — a 3B model imitating literal wording from a filled-in
# example — is preserved. What varies per persona here is which BEATS
# the example shows (e.g. Burnout's example shows "no new tasks
# assigned"; Consistent Learner's shows "no remedial framing"), which
# demonstrates persona-appropriate STRUCTURE without giving the model
# any real sentence to copy.

_FEW_SHOT_BANK = {
    "Burnout Pattern": (
        "###CONVERSATIONAL_REPLY###\n"
        "[reassurance-first opening; do not assign new tasks]\n\n"
        "###STUDENT_RECOMMENDATION###\n"
        "[one recovery-oriented micro-action]\n\n"
        "###EDUCATOR_RECOMMENDATION###\n"
        "[workload/wellness check-in]"
    ),
    "Anxiety-Spike Learner": (
        "###CONVERSATIONAL_REPLY###\n"
        "[validate the overwhelm first, then one grounding step]\n\n"
        "###STUDENT_RECOMMENDATION###\n"
        "[shorten next session, add a break]\n\n"
        "###EDUCATOR_RECOMMENDATION###\n"
        "[reassurance-first outreach, not a performance warning]"
    ),
    "Passive Watcher": (
        "###CONVERSATIONAL_REPLY###\n"
        "[offer a choice; connect material to real-world relevance]\n\n"
        "###STUDENT_RECOMMENDATION###\n"
        "[one small interactive action, not passive review]\n\n"
        "###EDUCATOR_RECOMMENDATION###\n"
        "[autonomy-supportive nudge]"
    ),
    "Silent Isolator": (
        "###CONVERSATIONAL_REPLY###\n"
        "[build rapport; invite them to share where they're stuck]\n\n"
        "###STUDENT_RECOMMENDATION###\n"
        "[one low-stakes social/forum interaction]\n\n"
        "###EDUCATOR_RECOMMENDATION###\n"
        "[proactive outreach — do not wait for them to initiate]"
    ),
    "Last-Minute Survivor": (
        "###CONVERSATIONAL_REPLY###\n"
        "[name the cost of cramming; propose spacing instead]\n\n"
        "###STUDENT_RECOMMENDATION###\n"
        "[concrete if-then micro-habit tied to the deadline]\n\n"
        "###EDUCATOR_RECOMMENDATION###\n"
        "[scaffolded reminder ahead of the next deadline]"
    ),
    "Consistent Learner": (
        "###CONVERSATIONAL_REPLY###\n"
        "[affirm what's working, then offer a stretch opportunity]\n\n"
        "###STUDENT_RECOMMENDATION###\n"
        "[one enrichment/advanced action — no remedial framing]\n\n"
        "###EDUCATOR_RECOMMENDATION###\n"
        "[peer-mentoring or leadership opportunity]"
    ),
}

_DEFAULT_FEW_SHOT = (
    "###CONVERSATIONAL_REPLY###\n"
    "[warm reply addressing the actual question]\n\n"
    "###STUDENT_RECOMMENDATION###\n"
    "[one grounded action for the student]\n\n"
    "###EDUCATOR_RECOMMENDATION###\n"
    "[one grounded action for the educator]"
)


def get_few_shot_example(persona: str) -> str:
    """Pure lookup — no computation, no fabrication."""
    return _FEW_SHOT_BANK.get(persona, _DEFAULT_FEW_SHOT)


# =========================================================
# 6. RETRIEVAL SAFETY CLASSIFICATION (SPRINT 8 ADDITION)
# =========================================================
# Hallucination Safety Hardening requirement: "Reasoning Layer — return
# one of SAFE / UNCERTAIN / NEEDS_REVIEW based on retrieval quality."
#
# This is deliberately a PURE function over primitives (not
# StudentContext, not a ConfidenceResult object) so this module stays
# the dependency-light function library described in the module
# docstring above — it does not need to import confidence.py or
# context_builder.py's retrieval-specific fields to do this
# classification; the caller (mentor_service.py) already has the three
# numbers this needs and passes them in directly.
#
# The three inputs are all ALREADY COMPUTED elsewhere:
#   - `max_similarity` comes straight from retrieval.py's per-chunk
#     `similarity_score` values (the strongest one seen this turn).
#   - `normalized_confidence` comes from confidence.py's
#     `ConfidenceResult.normalized_score`.
#   - `contradiction_detected` comes from confidence.py's
#     `ConfidenceResult.contradiction_detected`.
# This function only combines them into one categorical label; it does
# not recompute or reinterpret any of them.

# How far above the raw thresholds a value must sit before we call it
# fully SAFE rather than UNCERTAIN. A response that clears the bar by
# only a hair is still worth flagging as borderline rather than fully
# trusted — this margin is a heuristic tuning knob, not a statistical
# guarantee.
_SAFETY_MARGIN = 0.15


def classify_retrieval_safety(
    max_similarity: float,
    normalized_confidence: float,
    contradiction_detected: bool,
    similarity_threshold: float,
    confidence_threshold: float,
) -> str:
    """
    Returns one of "SAFE", "UNCERTAIN", "NEEDS_REVIEW":

      NEEDS_REVIEW - max_similarity is below similarity_threshold, OR
                     normalized_confidence is below confidence_threshold,
                     OR the retrieved evidence was flagged as
                     contradictory. This mirrors (and is consistent
                     with) confidence.py's own `needs_human_review`
                     logic, expressed here as a label rather than a
                     boolean so it can be surfaced to API consumers
                     alongside SAFE/UNCERTAIN as a three-way status.

      UNCERTAIN    - clears both thresholds, but only within
                     `_SAFETY_MARGIN` of one of them — evidence is
                     technically sufficient but thin enough that a
                     human reviewer may still want to spot-check it.

      SAFE         - comfortably clears both thresholds with no
                     contradiction detected.
    """
    if (
        max_similarity < similarity_threshold
        or normalized_confidence < confidence_threshold
        or contradiction_detected
    ):
        return "NEEDS_REVIEW"

    similarity_borderline = max_similarity < similarity_threshold * (1 + _SAFETY_MARGIN)
    confidence_borderline = normalized_confidence < confidence_threshold * (1 + _SAFETY_MARGIN)

    if similarity_borderline or confidence_borderline:
        return "UNCERTAIN"

    return "SAFE"
