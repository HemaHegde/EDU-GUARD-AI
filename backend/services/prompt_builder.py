"""
prompt_builder.py

Builds the system prompt for Aura as a series of clearly labeled
sections, instead of one monolithic f-string. This makes it possible to:
- omit a section cleanly when its data is unavailable (e.g. SHAP,
  cognitive state) instead of printing "None" into the prompt
- unit test each section independently
- extend with new sections later without touching unrelated ones

This module does not call the LLM and does not gather data — it only
formats a StudentContext (+ retrieved chunks + question) into a prompt.

SPRINT 2 UPDATE:
- The Cognitive State section is now a dedicated "Cognitive
  Intelligence" section that explicitly renders the fields produced by
  the real GRU cognitive pipeline (via cognitive_adapter.py /
  cognitive_service.py). Nothing here computes or invents any of these
  values — it only formats whatever context.cognitive_state already
  contains (or states plainly that it's unavailable).

SPRINT 2 BUGFIX (SHAP rendering):
- `_section_shap_summary` previously assumed `context.shap_explanation`
  was a flat {feature_name: importance_value} dict and iterated it
  directly with `.items()`. In reality, `shap_service.get_shap_explanation()`
  returns a NESTED dict shaped like:
      {
          "top_features": [ {feature, value, shap_value,
                              impact_direction, impact_magnitude}, ... ],
          "feature_importance": {feature_name: shap_value, ...},
          "summary": "<human-readable sentence>",
      }
  This was fixed to read the known keys explicitly rather than
  iterating blindly. No SHAP computation, no confidence logic, no
  persona logic, and no retrieval/cognitive logic were touched.

SPRINT 3 UPDATE (PROMPT EFFICIENCY OPTIMIZATION):
- Goal: cut total system-prompt size from ~9000 characters to roughly
  2500-3500 characters, to reduce qwen2.5:3b latency, WITHOUT changing
  any business logic: SHAP integration, persona mapping, cognitive
  intelligence, retrieval, confidence, output format, or API schema
  are all functionally identical to before.
- Every optimization below is a RENDERING/FORMATTING change only:
  fewer words, denser layout, and (for SHAP / retrieved material /
  cognitive recommendations) a cap on how many items are *displayed*
  to the LLM. No underlying computation, data source, schema, or
  function signature was altered. See the inline comments on each
  section for the specific rationale.
- Net effect measured on a typical fully-populated context: the
  combined section text drops from ~9000 characters to ~2500-3500
  characters, depending on chat history length and how many SHAP
  features / retrieved chunks are available for a given student.

SPRINT 5 UPDATE (persona-driven reasoning strengthened):
- Fixed a bug where "Consistent Learner" was missing from
  `_PSYCHOLOGICAL_FRAMEWORKS`, silently falling back to the generic
  default framework. Added `_PERSONA_ACTIONS` / `get_persona_actions`,
  a concrete per-persona {student, educator} action pair used both in
  the prompt and as a deterministic fallback (see mentor_service.py).

SPRINT 6 UPDATE (EVIDENCE PRIORITIZATION / MULTI-SIGNAL SYNTHESIS):
Previously, persona was surfaced as one line among several equally-
weighted evidence sections (risk, SHAP, cognitive, material). A 3B
model given six roughly-equal-weight sections tends to either default
to the first/most salient one (persona) or blend everything into
generic advice — which is exactly the reported symptom ("persona
mentioned but not driving reasoning", "almost every learner receives
similar responses").

This sprint adds one new, code-computed section: `_section_priority_focus`,
built by `determine_priority_focus()`. It does NOT introduce new data —
it inspects fields already present on `StudentContext` (persona,
risk_level, cognitive_state, shap_explanation) and applies a documented,
deterministic priority order to decide which ALREADY-AVAILABLE signal
should anchor this specific response, then tells the LLM explicitly:
"here is your one primary focus, and here is why." This converts an
implicit multi-source weighting problem (hard for a 3B model) into an
explicit instruction-following problem (something small models are
comparatively good at).

The priority order itself is not arbitrary — each rule is grounded in
why that signal would dominate psychologically (see inline comments on
`determine_priority_focus`), and several thresholds (confusion_score
>= 70, engagement_score < 50) deliberately reuse the exact thresholds
`cognitive_service.py`'s own recommendation engine already treats as
meaningful, rather than inventing new cutoffs.

`determine_priority_focus()` is also imported by mentor_service.py so
that the deterministic fallback recommendations (used when the LLM
misses the structured markers) reason about evidence the *same* way
the prompt does — keeping LLM-authored and fallback-authored
recommendations consistent with each other.
"""

from typing import List, Dict

from .context_builder import StudentContext
from .reasoning_layer import (
    build_evidence_profile,
    build_conversation_plan,
    compute_evidence_confidence,
    derive_persona_memory,
)

# SPRINT 7 UPDATE (multi-signal reasoning architecture):
# Adds four new, purely additive prompt sections built by
# reasoning_layer.py: Evidence Fusion, Conversation Plan, Continuity
# (persona memory / recommendation diversity), and Evidence Confidence
# weighting. See reasoning_layer.py's module docstring for the full
# rationale and hard constraints (no schema changes; Persona Memory
# and Recommendation Diversity are derived only from
# context.chat_history, which context_builder.py already builds from
# the existing mentor_history table).
#
# Nothing about the EXISTING sections, `determine_priority_focus()`,
# `get_persona_actions()`, `_PSYCHOLOGICAL_FRAMEWORKS`, the SHAP/
# cognitive/risk rendering, or `build_system_prompt`'s signature has
# changed in a breaking way -- the new sections are appended to the
# existing section list, and `_section_priority_focus` now takes the
# already-computed `focus` dict as a parameter (previously recomputed
# it internally) purely so `determine_priority_focus(context)` is
# called ONCE per prompt build instead of twice; its return value and
# every downstream consumer of it (fallback recommendations in
# mentor_service.py) are unchanged.

# =========================================================
# PSYCHOLOGICAL FRAMEWORK LOOKUP
# =========================================================
# Persona -> psychological framework mapping is UNCHANGED (same keys,
# same theory, same core strategy per persona, per Sprint 1 requirement
# #4: persona GUIDES the response, it does not by itself determine the
# recommendation). Only the WORDING was shortened (goal #3: shorten
# verbose wording without changing meaning) — each entry still names
# the theory and its core techniques, just as a single dense sentence
# instead of a multi-clause paragraph.

_PSYCHOLOGICAL_FRAMEWORKS = {
    "Burnout Pattern": (
        "CBT — workload reduction and recovery: break tasks into tiny "
        "steps, challenge all-or-nothing thinking, explicitly recommend "
        "reducing load this week and building in a recovery break; "
        "flag for educator outreach rather than pushing more output."
    ),
    "Passive Watcher": (
        "SDT — autonomy and interactive engagement: connect material to "
        "real-world relevance, offer a choice of next step, use "
        "Socratic questions, and nudge toward one small interactive "
        "action (a quiz, a discussion post) instead of passive review."
    ),
    "Anxiety-Spike Learner": (
        "Mindfulness — reassurance and pacing: validate the overwhelm "
        "first, use a grounding technique, recommend shorter study "
        "sessions with breaks, and reinforce confidence by naming a "
        "concrete recent strength before assigning anything new."
    ),
    "Silent Isolator": (
        "Social Presence Theory — connection: build rapport, actively "
        "invite them to ask a question or share where they're stuck, "
        "suggest one low-stakes social/forum interaction, and flag for "
        "proactive educator outreach given the isolation pattern."
    ),
    "Last-Minute Survivor": (
        "Implementation Intentions — planning: propose a concrete If-Then "
        "study micro-habit tied to a specific deadline, recommend spaced "
        "revision over cramming, and name the cognitive-load cost of "
        "leaving it to the last minute."
    ),
    "Consistent Learner": (
        "SDT — enrichment and growth: affirm what is already working, "
        "then offer a stretch opportunity — advanced material, a peer-"
        "mentoring or leadership role — rather than remedial advice; "
        "this student does not need risk-reduction framing."
    ),
}

_DEFAULT_FRAMEWORK = "General supportive mentoring."


def _get_psychological_framework(persona: str) -> str:
    return _PSYCHOLOGICAL_FRAMEWORKS.get(persona, _DEFAULT_FRAMEWORK)


def get_psychological_framework(persona: str) -> str:
    """
    SPRINT 7 ADDITION: public wrapper around `_get_psychological_framework`.
    mentor_service.py needs the same framework text prompt_builder.py
    uses (to build a conversation plan consistent with the prompt) but
    shouldn't reach into a private, underscore-prefixed helper across
    module boundaries. Pure passthrough — no new lookup table, no
    behaviour change.
    """
    return _get_psychological_framework(persona)


# =========================================================
# DETERMINISTIC, EVIDENCE-GROUNDED PERSONA ACTIONS
# =========================================================
# SPRINT 5 ADDITION.
#
# Two uses:
#   1. Fed into the prompt (see _section_persona) as a short, concrete
#      "lead with" action per persona, so the LLM has a specific,
#      persona-differentiated behaviour to open with instead of
#      reaching for generic "take deep breaths" / "study more" advice
#      (Problem #2 in the improvement brief).
#   2. Reused, unmodified, by mentor_service.py as the basis for a
#      deterministic fallback Student/Educator Recommendation when the
#      LLM does not emit the structured markers (Problem #3). This is
#      NOT a new invented data source — every fragment here restates
#      the same six persona archetypes already defined in
#      `_PSYCHOLOGICAL_FRAMEWORKS` above (and, for the educator side,
#      mirrors `intervention_map` in persona_service.py). No new
#      persona, feature, or dataset is introduced.
_PERSONA_ACTIONS = {
    "Burnout Pattern": {
        "student": "reduce your workload this week and schedule one real recovery break before your next study session",
        "educator": "flag for a workload/wellness check-in; consider a short extension or reduced load",
    },
    "Passive Watcher": {
        "student": "pick one interactive action today (a short quiz or a discussion post) instead of passive re-reading",
        "educator": "prompt with an autonomy-supportive nudge — offer a choice of task rather than a mandate",
    },
    "Anxiety-Spike Learner": {
        "student": "shorten your next study session and take a short break partway through to reduce pressure",
        "educator": "check in with a reassurance-first message; avoid framing this as a performance warning",
    },
    "Silent Isolator": {
        "student": "ask one question in the course forum or to a classmate this week, even a small one",
        "educator": "reach out proactively — this student is unlikely to initiate contact themselves",
    },
    "Last-Minute Survivor": {
        "student": "set a specific If-Then plan (e.g. 'after dinner, review one topic') to space out revision before the deadline",
        "educator": "consider a scaffolded reminder or interim checkpoint ahead of the next deadline",
    },
    "Consistent Learner": {
        "student": "take on an enrichment activity or advanced topic — current performance supports it",
        "educator": "consider this student for a peer-mentoring or leadership opportunity",
    },
}

_DEFAULT_PERSONA_ACTIONS = {
    "student": "review recent course material and reach out with any specific questions",
    "educator": "no strong behavioural pattern is on record yet; monitor and re-check after more activity data is available",
}


def get_persona_actions(persona: str) -> Dict[str, str]:
    """
    Public helper (also imported by mentor_service.py) returning the
    concrete {student, educator} action fragments for a persona, or a
    neutral default for personas outside the known set (e.g. "Unknown
    Persona"). Pure lookup — no computation, no fabrication.
    """
    return _PERSONA_ACTIONS.get(persona, _DEFAULT_PERSONA_ACTIONS)


# =========================================================
# EVIDENCE PRIORITIZATION (SPRINT 6)
# =========================================================
# Decides which ALREADY-AVAILABLE piece of evidence should be the
# single dominant focus of this response, instead of asking the LLM to
# implicitly weigh six co-equal sections itself. Every branch below
# only reads fields already present on StudentContext; nothing is
# computed, scored, or fabricated here that doesn't already exist
# upstream (risk_service, shap_service, cognitive_adapter,
# persona_service).
#
# ORDER RATIONALE (why this cascade, not another):
#   1. Consistent Learner + low/no risk is checked FIRST because it's
#      a hard behavioural requirement (never give remedial framing to
#      a student who isn't at risk) — it's a guardrail, not a "which
#      signal is strongest" comparison, so it takes precedence over
#      the signal-strength checks below it.
#   2. Confusion (Cognitive Load Theory) is checked before engagement:
#      a confused student cannot productively act on an "engage more"
#      nudge until the confusion itself is addressed — CLT treats
#      unresolved cognitive load as blocking any subsequent action.
#   3. Severe burnout (persona + High risk together, not persona
#      alone) is checked next because it is the closest thing this
#      system has to a safety-relevant signal — pushing more tasks at
#      a burned-out high-risk student is actively counterproductive.
#   4. Low engagement is next: it's actionable and immediate, but
#      lower-stakes than confusion or burnout.
#   5. SHAP-identified inactivity is checked last among the overrides
#      because it's a slower-moving behavioural pattern (days-scale)
#      versus the more acute cognitive/persona signals above it.
#   6. If nothing crosses a threshold, the persona's own strategy
#      (already the richest single signal) carries the response — this
#      is the same behaviour as before Sprint 6 for a "quiet" evidence
#      profile, not a regression.
_CONFUSION_THRESHOLD = 70   # matches cognitive_service.py's own >70 "strong confusion" rule
_ENGAGEMENT_THRESHOLD = 50  # matches cognitive_service.py's own <50 "gamified exercises" rule


def determine_priority_focus(context: "StudentContext") -> Dict[str, str]:
    """
    Returns a dict:
        {
            "id": short signal identifier (for logging/debugging only),
            "instruction": one sentence telling the LLM what to anchor on and why,
            "reason": short evidence citation (which field, which value),
            "student_action": concrete student-facing action for this signal,
            "educator_action": concrete educator-facing action for this signal,
        }

    Deterministic and side-effect-free. Reused by mentor_service.py for
    fallback recommendations so that LLM-authored and fallback-authored
    output reason about evidence consistently.
    """

    persona = context.persona
    risk_level = context.risk_level
    persona_actions = get_persona_actions(persona)

    cognitive = context.cognitive_state if context.cognitive_state_available else None

    shap_top = None
    if context.shap_available and context.shap_explanation:
        top_features = (context.shap_explanation or {}).get("top_features") or []
        if top_features:
            shap_top = top_features[0]

    # --- 1. Guardrail: Consistent Learner + non-elevated risk -> enrichment, never remediation ---
    if persona == "Consistent Learner" and risk_level in ("Low", "Unknown", None):
        return {
            "id": "enrichment",
            "instruction": (
                "This learner is stable and not at elevated risk — lead "
                "with enrichment or a leadership opportunity. Do not use "
                "remedial or risk-reduction framing anywhere in your reply."
            ),
            "reason": f"persona=Consistent Learner, risk_level={risk_level}",
            "student_action": persona_actions["student"],
            "educator_action": persona_actions["educator"],
        }

    # --- 2. Confusion (Cognitive Load Theory: resolve blocking load first) ---
    if cognitive and cognitive.get("confusion_score") is not None and cognitive["confusion_score"] >= _CONFUSION_THRESHOLD:
        return {
            "id": "confusion",
            "instruction": (
                "Confusion is the dominant signal for this response — "
                "slow down and clarify the underlying concept before "
                "suggesting any action step; do not add unrelated advice."
            ),
            "reason": f"cognitive confusion_score={cognitive['confusion_score']} (>= {_CONFUSION_THRESHOLD})",
            "student_action": (
                "ask for the specific step or concept that's unclear to "
                "be broken down before moving on to anything new"
            ),
            "educator_action": (
                "consider re-explaining this topic or providing a worked "
                "example — confusion score is elevated"
            ),
        }

    # --- 3. Severe burnout: persona AND High risk together, not persona alone ---
    if persona == "Burnout Pattern" and risk_level == "High":
        return {
            "id": "burnout",
            "instruction": (
                "Burnout risk is severe — prioritize workload reduction "
                "and recovery over anything else; do not assign new work."
            ),
            "reason": "persona=Burnout Pattern, risk_level=High",
            "student_action": persona_actions["student"],
            "educator_action": persona_actions["educator"],
        }

    # --- 4. Low engagement -> encourage active interaction ---
    if cognitive and cognitive.get("engagement_score") is not None and cognitive["engagement_score"] < _ENGAGEMENT_THRESHOLD:
        return {
            "id": "engagement",
            "instruction": (
                "Engagement is the dominant signal — the priority is one "
                "small active/interactive step, not passive review."
            ),
            "reason": f"cognitive engagement_score={cognitive['engagement_score']} (< {_ENGAGEMENT_THRESHOLD})",
            "student_action": (
                "do one small interactive action now (answer a quiz "
                "question, post one comment) instead of continuing to "
                "watch or read passively"
            ),
            "educator_action": (
                "prompt with an interactive task rather than more "
                "reading or video material — engagement score is low"
            ),
        }

    # --- 5. SHAP-identified inactivity as the strongest measured driver ---
    if shap_top and shap_top.get("impact_direction") == "positive" and "inactiv" in str(shap_top.get("feature", "")).lower():
        return {
            "id": "inactivity",
            "instruction": (
                "Inactivity is the strongest measured risk driver — "
                "anchor advice on resuming regular activity, not "
                "general study tips."
            ),
            "reason": f"SHAP top feature='{shap_top.get('feature')}', direction=+risk",
            "student_action": (
                "log in and complete one small piece of coursework "
                "today to break the inactivity streak, then build back "
                "up gradually"
            ),
            "educator_action": (
                "reach out directly — inactivity is the strongest "
                "measured driver of this student's risk"
            ),
        }

    # --- 6. Default: no single signal dominates, persona strategy carries the response ---
    return {
        "id": "persona_default",
        "instruction": (
            f"No single signal dominates this evidence set — let the "
            f"{persona} strategy above guide tone and focus."
        ),
        "reason": "no cognitive/persona/SHAP threshold was crossed",
        "student_action": persona_actions["student"],
        "educator_action": persona_actions["educator"],
    }


# =========================================================
# DISPLAY CAPS (prompt-rendering limits only — see rationale per section)
# =========================================================
# These constants control ONLY how much of an already-computed result
# is shown to the LLM inside the prompt. They do not change what is
# computed, stored, or returned anywhere else in the system (e.g. the
# full SHAP ranking remains available via shap_service/the API; the
# full retrieved-chunk list is still returned by retrieval.py; the full
# six-field cognitive state remains in context.cognitive_state exactly
# as produced by cognitive_service.py).

_MAX_SHAP_FEATURES_IN_PROMPT = 3   # goal #5: preserve TOP contributing features, drop the rest from the prompt text only
_MAX_RECOMMENDATIONS_IN_PROMPT = 2  # goal #6: show the most actionable cognitive tips, not the full list, in the prompt
_MAX_CHUNKS_IN_PROMPT = 2          # goal #4: use only the most relevant retrieved chunks (already relevance-ranked by retriever.retrieve())
_MAX_CHARS_PER_CHUNK = 350         # goal #4: cap each chunk's contribution
_MAX_TOTAL_MATERIAL_CHARS = 800    # goal #4: hard cap on the combined learning-material section

# SPRINT 4 (history-driven prompt bloat fix): chat_history, as built by
# context_builder.py, is the FULL formatted history (up to
# history_limit turns fetched from the DB, already oldest -> newest).
# It was previously rendered into the prompt in full, which is what
# made History balloon to ~5800+ characters over a longer
# conversation. These caps limit ONLY how much of that already-built
# string is echoed into THIS prompt; context.chat_history itself,
# context_builder.py, and the DB query/limit behind it are untouched.
_MAX_HISTORY_TURNS = 4             # keep only the most recent N conversational turns
_MAX_HISTORY_CHARS = 1100          # hard cap applied after turn-limiting, keeping the most recent tail
                                    # (set slightly below the 1200-char target to leave headroom for the
                                    # "History:\n" label and, when applicable, the truncation notice line)
_HISTORY_TRUNCATION_NOTICE = "[Earlier conversation omitted for brevity]"


# =========================================================
# SECTION BUILDERS
# =========================================================

def _section_system_role() -> str:
    # goal #3: same coverage (academics, motivation, stress,
    # productivity, career guidance, coding, general knowledge, normal
    # conversation) as one dense sentence instead of three lines.
    return (
        "You are Aura — a warm, emotionally supportive AI mentor for "
        "academics, motivation, stress, productivity, career guidance, "
        "coding, general knowledge, and everyday conversation."
    )


def _section_student_profile(context: StudentContext) -> str:
    return f"Student: {context.student_name}"


def _section_academic_risk(context: StudentContext) -> str:
    # goal #3: identical information (risk level, risk score, behaviour
    # summary), rendered as one compact line instead of three labeled
    # lines. The "do not invent risk" instruction is preserved verbatim
    # in spirit when risk_score is None.
    if context.risk_score is None:
        return "Academic Risk: not available. Do not state or imply a risk level."
    return (
        f"Academic Risk: {context.risk_level} (score {context.risk_score}). "
        f"Behaviour: {context.behaviour_summary}"
    )


def _format_shap_feature_line(feature_item: Dict) -> str:
    """
    Renders a single entry from shap_service's `top_features` list as
    a short, comma-free fragment suitable for inline joining (rather
    than a full bullet line), e.g.:
        "inactivity_days (+risk, 0.42)"

    Each entry is expected to look like (see explainability_service.py /
    shap_service.py):
        {
            "feature": "inactivity_days",
            "value": 12.0,
            "shap_value": 0.42,
            "impact_direction": "positive" | "negative",
            "impact_magnitude": 0.42,
        }

    We pull out named fields and format them as plain text — we never
    str()/repr() the dict itself. Direction labels are shortened
    ("+risk"/"-risk"/"unclear") but carry the same meaning as the
    previous "increasing risk"/"decreasing risk"/"direction unknown"
    phrasing (goal #3).
    """
    feature_name = feature_item.get("feature", "unknown_feature")
    shap_value = feature_item.get("shap_value")
    direction = feature_item.get("impact_direction")

    if direction == "positive":
        direction_phrase = "+risk"
    elif direction == "negative":
        direction_phrase = "-risk"
    else:
        direction_phrase = "unclear"

    if shap_value is None:
        return f"{feature_name} ({direction_phrase})"

    return f"{feature_name} ({direction_phrase}, {shap_value:.2f})"


def _section_shap_summary(context: StudentContext) -> str:
    """
    Renders the SHAP explanation section from the structured payload
    produced by shap_service.get_shap_explanation():
        {
            "top_features": [ {feature, value, shap_value,
                                impact_direction, impact_magnitude}, ... ],
            "feature_importance": {feature_name: shap_value, ...},
            "summary": "<human-readable sentence>",
        }

    goal #5 (compress SHAP rendering while preserving top contributing
    features): only the top `_MAX_SHAP_FEATURES_IN_PROMPT` entries from
    `top_features` are rendered, joined inline instead of one bullet
    per line. This is a display cap only — shap_service still computes
    and returns the full ranking; nothing about SHAP computation,
    schema, or the `feature_importance` payload is touched.

    Backward compatibility with a flat {feature: value} dict is
    preserved, also capped and compacted for consistency.
    """
    if not context.shap_available or not context.shap_explanation:
        return "SHAP: not available. Do not invent specific risk drivers."

    shap_data = context.shap_explanation

    # --- Current / expected schema: nested dict with top_features + summary ---
    if isinstance(shap_data, dict) and "top_features" in shap_data:
        summary = shap_data.get("summary") or "No summary available."
        top_features = (shap_data.get("top_features") or [])[:_MAX_SHAP_FEATURES_IN_PROMPT]

        if top_features:
            feature_fragment = "; ".join(_format_shap_feature_line(item) for item in top_features)
        else:
            feature_fragment = "no individual drivers available"

        return f"SHAP drivers (top): {feature_fragment}. Summary: {summary}"

    # --- Backward-compatible fallback: flat {feature: importance} dict ---
    if isinstance(shap_data, dict):
        items = list(shap_data.items())[:_MAX_SHAP_FEATURES_IN_PROMPT]
        feature_fragment = "; ".join(f"{feature}={value}" for feature, value in items)
        return f"SHAP drivers (top): {feature_fragment}"

    # --- Last-resort fallback: unexpected shape, do not dump raw repr ---
    return "SHAP: available but in an unrecognized format. Do not invent risk drivers."


def _section_persona(context: StudentContext) -> str:
    # goal #3: same three data points (persona, intervention style,
    # strategy), rendered on one line instead of three.
    #
    # SPRINT 5 ADDITION: a fourth data point, `lead_action`, is now
    # appended. This is the concrete "student" fragment from
    # `_PERSONA_ACTIONS` (see rationale above _PERSONA_ACTIONS) — it
    # gives the LLM one specific, persona-differentiated behaviour to
    # visibly act on, rather than only a theory label it can ignore.
    # This directly targets Problem #1 (persona mentioned but not
    # driving reasoning) and Problem #4 (six personas should behave
    # like six different mentors) from the improvement brief.
    framework = _get_psychological_framework(context.persona)
    lead_action = get_persona_actions(context.persona)["student"]
    return (
        f"Persona: {context.persona} ({context.intervention_style}). "
        f"Strategy: {framework} Lead with: {lead_action}."
    )


def _section_cognitive_intelligence(context: StudentContext) -> str:
    """
    Cognitive Intelligence section, sourced strictly from the real
    GRU-based pipeline (context.cognitive_state, populated by
    cognitive_adapter.py -> cognitive_service.py). If unavailable, says
    so plainly — never infers or fabricates a cognitive/emotional state.

    goal #6 (compress to fields actually useful for mentoring): this
    section renders focus_score, engagement_score, and confusion_score
    (the three signals that directly inform conversational tone/pacing)
    plus up to `_MAX_RECOMMENDATIONS_IN_PROMPT` system-generated tips.

    `ai_risk_probability` and `learning_velocity` are intentionally
    OMITTED FROM THIS PROMPT SECTION ONLY:
      - `ai_risk_probability` duplicates information already surfaced
        in the Academic Risk section above (which comes from the
        OULAD-trained XGBoost model), so repeating it here would be a
        duplicated instruction/fact within the same prompt (goal #2)
        without giving the LLM new signal for tone-setting.
      - `learning_velocity` is a finer-grained pacing/trend metric more
        relevant to instructor-facing analytics than to a single
        conversational turn's tone.
    IMPORTANT: this is a prompt-formatting decision only. Neither field
    is removed from computation, storage, or the API — both remain
    fully present, unmodified, in context.cognitive_state exactly as
    produced by the GRU cognitive-state model described in the paper
    (Section IV.E); they are simply not echoed into this particular
    LLM prompt.
    """
    if not context.cognitive_state_available or not context.cognitive_state:
        return (
            "Cognitive state: not available. Do not infer or fabricate "
            "focus, engagement, or confusion level."
        )

    state = context.cognitive_state

    recommendations = state.get("recommendations") or []
    if isinstance(recommendations, (list, tuple)):
        capped = list(recommendations)[:_MAX_RECOMMENDATIONS_IN_PROMPT]
        recs_fragment = "; ".join(str(item) for item in capped) if capped else "none"
    else:
        recs_fragment = str(recommendations)

    return (
        f"Cognitive state: focus={state.get('focus_score')}, "
        f"engagement={state.get('engagement_score')}, "
        f"confusion={state.get('confusion_score')}. Tips: {recs_fragment}"
    )


def _section_learning_material(retrieved_chunks: List[Dict[str, str]]) -> str:
    """
    goal #4: use only the most relevant retrieved chunks instead of
    concatenating all of them. `retrieved_chunks` is already ranked by
    relevance by retriever.retrieve() (untouched, unchanged) — this
    function only limits how many of those already-ranked chunks, and
    how much of each, get embedded into the prompt text.
    """
    if not retrieved_chunks:
        return "Learning material: none retrieved for this query."

    top_chunks = retrieved_chunks[:_MAX_CHUNKS_IN_PROMPT]
    trimmed_pieces = [item["chunk"][:_MAX_CHARS_PER_CHUNK] for item in top_chunks]
    combined = " ".join(trimmed_pieces)[:_MAX_TOTAL_MATERIAL_CHARS]
    return f"Learning material: {combined}"


def _section_conversation_history(chat_history: str) -> str:
    """
    SPRINT 4 (history-driven prompt bloat fix): renders only the most
    recent conversational turns instead of the entire chat_history
    string, to stop History from dominating total prompt size on
    longer conversations (observed: 5781 chars, >73% of the prompt).

    This is a rendering-only change:
    - `chat_history` itself (produced by context_builder.py, which is
      untouched) is not modified.
    - StudentContext, the DB query, and history_limit behaviour are
      untouched — the full fetched history is still passed in here;
      this function simply chooses how much of it to echo into the
      LLM prompt.

    Strategy (keeps whichever preserves continuity best, per spec):
    1. Split into individual turns and keep only the last
       `_MAX_HISTORY_TURNS` (context_builder.py formats each turn as
       "Student: <q>\\nAura: <a>\\n" and joins turns with "\\n", so each
       turn reliably starts with "Student: " — this split relies only
       on that existing, unchanged formatting convention).
    2. If the turn-limited result is still over `_MAX_HISTORY_CHARS`,
       trim further from the front (keeping the most recent tail).
    3. If anything was cut in either step, prepend the required
       "[Earlier conversation omitted for brevity]" notice so the
       model knows context was shortened.
    """
    if not chat_history:
        return "History: start of the conversation."

    raw_turns = chat_history.split("Student: ")
    turns = [f"Student: {turn}".strip("\n") for turn in raw_turns if turn.strip()]

    truncated = False

    if len(turns) > _MAX_HISTORY_TURNS:
        turns = turns[-_MAX_HISTORY_TURNS:]
        truncated = True

    trimmed_history = "\n".join(turns)

    if len(trimmed_history) > _MAX_HISTORY_CHARS:
        trimmed_history = trimmed_history[-_MAX_HISTORY_CHARS:]
        truncated = True

    if truncated:
        return f"History:\n{_HISTORY_TRUNCATION_NOTICE}\n{trimmed_history}"

    return f"History:\n{trimmed_history}"


def _section_priority_focus(focus: Dict[str, str]) -> str:
    """
    SPRINT 6 ADDITION. Renders the single dominant-signal decision from
    `determine_priority_focus()` as an explicit, high-salience prompt
    section. Placed immediately before the final instructions (not
    interleaved with the raw evidence sections above) so it reads as a
    synthesis/verdict over everything already presented, rather than
    just one more co-equal fact — and so it benefits from recency,
    which measurably helps small instruction-followers weight it.

    SPRINT 7 CHANGE: now takes the already-computed `focus` dict as a
    parameter instead of calling `determine_priority_focus(context)`
    itself. This is purely so `build_system_prompt` can compute
    `focus` once and share it with the new Sprint 7 sections
    (Evidence Fusion, Conversation Plan) below — the function/value
    `determine_priority_focus` produces is completely unchanged.
    """
    return (
        f"PRIORITY FOCUS: {focus['instruction']} "
        f"(basis: {focus['reason']})."
    )


def _section_evidence_fusion(context: StudentContext, focus: Dict[str, str]) -> str:
    """
    SPRINT 7 ADDITION. Renders `reasoning_layer.build_evidence_profile()`
    as an explicit PRIMARY / SECONDARY / SUPPORTING / lower-priority
    ranking, so the model builds its response in that order instead of
    blending unrelated evidence together. PRIMARY always matches the
    PRIORITY FOCUS section above exactly (same `focus` dict, same
    source of truth) — this section adds ranking over what's LEFT
    after that primary signal, it never overrides it.
    """
    profile = build_evidence_profile(context, focus)

    lines = [f"PRIMARY SIGNAL: {profile['primary']['label']} ({profile['primary']['reason']})."]
    if profile["secondary"]:
        lines.append(f"SECONDARY SIGNAL: {profile['secondary']['label']}.")
    if profile["supporting"]:
        lines.append(f"SUPPORTING SIGNAL: {profile['supporting']['label']}.")
    if profile["ignored"]:
        ignored_labels = ", ".join(item["label"] for item in profile["ignored"])
        lines.append(f"LOWER PRIORITY THIS TURN (mention only if directly asked): {ignored_labels}.")

    return "EVIDENCE FUSION:\n" + " ".join(lines)


def _section_conversation_plan(context: StudentContext, focus: Dict[str, str]) -> str:
    """
    SPRINT 7 ADDITION. Renders `reasoning_layer.build_conversation_plan()`
    — a concrete opening/teaching/recommendation/educator plan derived
    from the same PRIORITY FOCUS — so the model follows a plan instead
    of inventing response structure from scratch each turn.
    """
    persona_actions = get_persona_actions(context.persona)
    framework_text = _get_psychological_framework(context.persona)
    plan = build_conversation_plan(context, focus, persona_actions, framework_text)

    return (
        "CONVERSATION PLAN: "
        f"Opening goal: {plan['opening_goal']}. "
        f"Teaching style: {plan['teaching_style']} "
        f"Recommendation style: {plan['recommendation_style']}. "
        f"Educator goal: {plan['educator_goal']}."
    )


def _section_continuity(context: StudentContext) -> str:
    """
    SPRINT 7 ADDITION (Persona Memory + Recommendation Diversity).
    Renders `reasoning_layer.derive_persona_memory()`, which reads
    ONLY `context.chat_history` (already built by context_builder.py
    from the existing mentor_history table — no new query, no new
    storage). Tells the model what NOT to repeat, rather than handing
    it a new "memory" fact to reason about.
    """
    memory = derive_persona_memory(context.chat_history)

    if not memory["has_history"]:
        return "CONTINUITY: start of the conversation — nothing prior to avoid repeating."

    pieces = []
    if memory["previous_opening"]:
        pieces.append(
            f"do not reopen with wording similar to: \"{memory['previous_opening']}...\""
        )
    if memory["used_action_tags"]:
        used = ", ".join(memory["used_action_tags"])
        pieces.append(f"you already suggested actions like: {used} — offer something different this turn")

    if not pieces:
        pieces.append("no specific repeat detected, but still vary phrasing from prior turns")

    return "CONTINUITY: " + "; ".join(pieces) + "."


def _section_confidence_weighting(context: StudentContext) -> str:
    """
    SPRINT 7 ADDITION (Evidence Confidence Weighting). Renders
    `reasoning_layer.compute_evidence_confidence()` — heuristic 0-1
    weights per evidence source (how decisive/available each signal
    is), not a replacement for confidence.py's High/Medium/Low
    response-level rubric, which still governs the `confidence` field
    returned by the API, unchanged.
    """
    weights = compute_evidence_confidence(context)
    return (
        "EVIDENCE CONFIDENCE (0=unavailable/weak, 1=strong/decisive): "
        f"risk={weights['risk']}, persona={weights['persona']}, "
        f"shap={weights['shap']}, cognitive={weights['cognitive']}. "
        "Lean your reasoning toward the higher-confidence signals."
    )


def _section_response_instructions() -> str:
    # goal #2 + #3 (Sprint 3), further compressed in Sprint 4 to target
    # ~250-300 characters. Every distinct constraint from the original
    # 13-point list is still present, just as terse fragments instead
    # of full sentences: warm tone + assigned strategy; evidence-based
    # substance (not persona alone); risk-mention caution; never reveal
    # internals/raw scores; never invent unavailable data; adapt style
    # to question type; stay concise; keep recommendations grounded.
    #
    # SPRINT 5 ADDITION: two clauses added, both directly targeting the
    # failure modes in the improvement brief and the AI Mentor Failure
    # Analysis (repeated "Partial Success" from missing structured
    # markers, and near-identical advice across personas):
    #   - an explicit ban on generic filler advice ("take deep breaths",
    #     "study more", "stay positive") unless tied to the assigned
    #     persona strategy and at least one concrete evidence point —
    #     this is what actually forces persona to drive the reasoning,
    #     not just be mentioned (Problem #1, #2, #5).
    #   - a blunt, repeated reminder that the three section markers are
    #     mandatory and must not be skipped, dropped, or renamed —
    #     placed at the END of the prompt (recency helps small models
    #     follow instructions) in addition to the format block appended
    #     by mentor_service.py (Problem #3).
    #
    # SPRINT 6 ADDITION: three more clauses, targeting evidence
    # prioritization, reply/recommendation consistency, and wording
    # diversity specifically:
    #   - explicitly points the model at the PRIORITY FOCUS section
    #     above and requires both recommendations to follow from it,
    #     rather than treating all evidence as equally weighted (this
    #     is the instruction-side half of Sprint 6; the code-side half
    #     is `determine_priority_focus` deciding WHAT that focus is).
    #   - requires the two recommendations to be consistent with (not
    #     introduce new, unrelated advice beyond) the conversational
    #     reply — addresses reported inconsistency between reply and
    #     recommendations.
    #   - a short wording-diversity instruction, since small models
    #     default to the same stock openers ("I understand...",
    #     "It sounds like...") turn after turn.
    # SPRINT 7 ADDITION: two more clauses, pointing the model at the
    # new synthesis sections rather than leaving it to notice them on
    # its own — same "explicit instruction-following beats implicit
    # weighting" rationale as Sprint 6's PRIORITY FOCUS clause:
    #   - follow the CONVERSATION PLAN's opening/teaching/recommendation
    #     structure instead of inventing response structure per turn.
    #   - honor CONTINUITY (do not reuse the previous opening phrase or
    #     the previously-used recommendation tags) — this is the
    #     instruction-side half of Recommendation Diversity; the
    #     code-side half is `derive_persona_memory()` deciding WHAT was
    #     already used.
    return (
        "Instructions: Be warm, follow the assigned strategy and the "
        "PRIORITY FOCUS above — that is the ONE signal your reply and "
        "both recommendations must anchor on; treat SECONDARY/"
        "SUPPORTING evidence in EVIDENCE FUSION as background only, and "
        "the LOWER PRIORITY items only if directly asked. Follow the "
        "CONVERSATION PLAN's opening goal, teaching style, and "
        "recommendation style rather than inventing your own structure. "
        "Respect CONTINUITY: do not reopen with the same phrasing as "
        "last time, and do not repeat a previously-used recommendation "
        "— offer a genuinely different one. Do not give generic advice "
        "(e.g. 'take deep breaths', 'study more', 'stay positive') "
        "unless tied to the priority focus. Both recommendations must "
        "follow directly from what your reply just said, not introduce "
        "new, unrelated advice. Mention risk only if relevant; never "
        "reveal raw scores or internals unless asked; never invent "
        "unavailable data. Vary your opening phrase — do not default "
        "to 'I understand' or 'It sounds like' every time. Adapt to "
        "question type; stay concise; keep recs evidence-based or say "
        "evidence is thin. You MUST output all three section markers "
        "below, exactly as given, every time — do not omit or rename "
        "them."
    )


# =========================================================
# PUBLIC ENTRY POINT
# =========================================================

# =========================================================
# TEMPORARY DEBUGGING (remove before shipping)
# =========================================================
# Prints the character length of each individual prompt section, plus
# the total, right before the assembled prompt is returned. This is
# diagnostic output only: it does not alter any section's text, does
# not change the join logic, and has no effect on what gets sent to
# the LLM. Safe to delete this block (and the _debug_print_section_sizes
# call below) once no longer needed.

_DEBUG_SECTION_LABELS = [
    "System Role",
    "Student Profile",
    "Academic Risk",
    "SHAP",
    "Persona",
    "Cognitive",
    "Learning Material",
    "History",
    "Priority Focus",
    "Evidence Fusion",       # Sprint 7
    "Conversation Plan",     # Sprint 7
    "Continuity",            # Sprint 7
    "Confidence Weighting",  # Sprint 7
    "Instructions",
]


def _debug_print_section_sizes(labeled_sections: List[tuple]) -> None:
    """Print a fixed-width report of each section's length and the total."""
    label_width = max(len(label) for label, _ in labeled_sections)
    total = sum(len(text) for _, text in labeled_sections)

    print("===== Prompt Section Sizes =====")
    for label, text in labeled_sections:
        dots = "." * (label_width - len(label) + 8)
        print(f"{label} {dots} {len(text)}")
    dots = "." * (label_width - len("TOTAL") + 8)
    print(f"TOTAL {dots} {total}")
    print("================================")


def build_system_prompt(
    context: StudentContext,
    retrieved_chunks: List[Dict[str, str]],
) -> str:
    """
    Assembles the full structured system prompt from individual,
    independently-testable sections. Section order is UNCHANGED through
    `_section_priority_focus` (Sprint 6). SPRINT 7 appends four new
    sections immediately after it — Evidence Fusion, Conversation Plan,
    Continuity, and Confidence Weighting — all synthesis-over-evidence
    sections, so they sit together after the raw evidence and right
    before the final instructions, same placement rationale as
    Sprint 6's Priority Focus (recency helps small instruction-
    followers weight a synthesis section over raw facts).

    `determine_priority_focus(context)` is now called ONCE here and
    the result (`focus`) is shared with `_section_priority_focus`,
    `_section_evidence_fusion`, and `_section_conversation_plan` —
    previously `_section_priority_focus` computed it internally. The
    function and its return value are unchanged; this is purely to
    avoid recomputing the same deterministic result multiple times
    per prompt build.
    """

    focus = determine_priority_focus(context)

    sections = [
        _section_system_role(),
        _section_student_profile(context),
        _section_academic_risk(context),
        _section_shap_summary(context),
        _section_persona(context),
        _section_cognitive_intelligence(context),
        _section_learning_material(retrieved_chunks),
        _section_conversation_history(context.chat_history),
        _section_priority_focus(focus),
        _section_evidence_fusion(context, focus),
        _section_conversation_plan(context, focus),
        _section_continuity(context),
        _section_confidence_weighting(context),
        _section_response_instructions(),
    ]

    # TEMPORARY DEBUGGING: report each section's size before returning.
    # No section text is modified here — this only reads len() on the
    # exact same strings that get joined below.
    _debug_print_section_sizes(list(zip(_DEBUG_SECTION_LABELS, sections)))

    return "\n\n".join(sections)
