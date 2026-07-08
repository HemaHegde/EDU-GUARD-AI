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

SPRINT 8 UPDATE (HALLUCINATION SAFETY HARDENING):
Three additive changes, none of which touch existing sections, the
priority-focus/evidence-fusion/conversation-plan logic, persona
mapping, or `build_system_prompt`'s existing section order up through
`_section_response_instructions`:

  1. `_section_learning_material` now renders each retrieved chunk's
     `source`, `chunk_id`, and `similarity_score` (all already
     computed by retrieval.py -- nothing new is calculated here) inline
     with the chunk text, and instructs the model to cite ONLY those
     exact source names. This directly supports the Evidence Citations
     and Retrieval Safety requirements.
  2. A new `select_chunks_for_prompt()` public helper factors out the
     "which chunks actually get shown to the LLM" slicing logic that
     used to live only inside `_section_learning_material`.
     mentor_service.py imports this so the "Sources:" list it appends
     to every answer matches EXACTLY what the model was shown -- never
     a larger or different set than what actually grounded the answer.
  3. A new `_section_hallucination_safety()` section is appended to
     `build_system_prompt`'s section list (after Learning Material,
     before the final Response Instructions) with an explicit,
     mandatory instruction: never fabricate, never answer outside the
     retrieved evidence for content questions, and say "I don't know"
     / defer to the fixed insufficient-evidence message when evidence
     is thin. This is a defense-in-depth prompt-level backstop --
     mentor_service.py's SIMILARITY_THRESHOLD gate already prevents the
     LLM from being called at all when evidence is clearly too weak;
     this section covers the borderline cases where the LLM is still
     called but should still decline to speculate on any part of the
     question the retrieved material doesn't actually cover.
"""

import re
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
# PHASE 3.3: ADAPTIVE CONVERSATION PLANNING
# =========================================================
# `determine_conversation_goal()` decides the CONVERSATIONAL GOAL for
# this turn -- i.e. what Aura's reply should be trying to accomplish
# ("reduce pressure", "clarify understanding", "encourage engagement"
# ...) -- BEFORE the response is generated. This is explicitly NOT a
# recommendation-selection decision: `recommendation_engine.py`,
# `select_recommendation()`, and `determine_priority_focus()` above are
# untouched and keep deciding WHAT concrete action is recommended. The
# conversation goal only decides the higher-level conversational
# OBJECTIVE the reply (and, optionally, the ordering of existing
# recommendation-reasoning sentence slots — see
# recommendation_reasoning.apply_conversation_goal_ordering) should
# serve.
#
# WHY DETERMINISTIC, NOT ML: same discipline as `determine_priority_focus`
# and every classifier in intent_classifier.py -- this is a plain,
# ordered set of dict/string comparisons over signals that ALREADY
# exist by the time this runs (persona, the already-computed priority
# focus, detected_intent, detected_emotion, detected_emotion_intensity,
# detected_emotion_trajectory). No new evidence is gathered, no model
# is called, and no randomness is involved, so identical inputs always
# yield the identical goal string -- fully auditable and unit-testable
# as a fixed table of input -> output pairs.
#
# PRECEDENCE (checked in this fixed order, most urgent/safety-relevant
# signal first -- mirroring the same "urgent signal first" precedence
# rationale already used by `classify_emotion`, `classify_intent`, and
# `determine_priority_focus` above):
#   1. Escalating emotional distress across recent turns is the single
#      most urgent signal available -- it overrides persona/intent/
#      focus framing, because no conversational objective matters if
#      the learner's emotional state is actively worsening.
#   2. Confusion as the dominant priority focus (Cognitive Load Theory,
#      same rationale as `determine_priority_focus`'s own ordering) --
#      an unresolved blocking concept must be clarified before any
#      other conversational objective can land.
#   3. Persona-level patterns with an unambiguous, single best-fit
#      conversational objective (Burnout Pattern, Passive Watcher,
#      Silent Isolator, Last-Minute Survivor, Consistent Learner).
#   4. Career intent, when nothing more urgent above already applies.
#   5. Success emotion, when nothing more urgent above already applies.
#   6. A positive, stable trajectory across recent turns.
#   7. Default: "General Support" -- no single signal above dominates,
#      so no more specific objective than general support is claimed.
_STABILIZING_TRAJECTORIES = {"Escalating Distress"}

_PERSONA_CONVERSATION_GOALS = {
    "Burnout Pattern": "Reduce pressure",
    "Passive Watcher": "Encourage engagement",
    "Silent Isolator": "Build connection",
    "Last-Minute Survivor": "Build planning habit",
    "Consistent Learner": "Encourage growth",
}

_DEFAULT_CONVERSATION_GOAL = "General Support"


def determine_conversation_goal(
    persona: str,
    priority_focus: Dict[str, str],
    detected_intent: str = "general",
    detected_emotion: str = "neutral",
    detected_emotion_intensity: str = "Medium",
    detected_emotion_trajectory: str = "neutral",
) -> str:
    """
    Returns a short conversational-goal string (e.g. "Reduce pressure",
    "Clarify understanding", "Stabilize emotion first"). Deterministic
    and side-effect-free -- uses ONLY the six already-computed inputs
    listed in the signature; it gathers no new evidence, calls no
    model, and has no randomness.

    `priority_focus` is the SAME dict `determine_priority_focus(context)`
    already produces (its "id" key is read here) -- callers should pass
    the one they already computed for this turn rather than calling
    `determine_priority_focus` a second time with different context.

    This never selects a recommendation, never reorders retrieval, and
    is never itself returned to an API consumer -- it is consumed only
    by `build_system_prompt`'s new Conversation Goal section and,
    optionally, by `recommendation_reasoning.py` to reorder (never
    change the content of) existing conversation-strategy slots.
    """
    focus_id = (priority_focus or {}).get("id", "")
    trajectory = (detected_emotion_trajectory or "").strip()

    # 1. Escalating distress overrides everything else.
    if trajectory in _STABILIZING_TRAJECTORIES:
        return "Stabilize emotion first"

    # 2. Confusion as the dominant priority focus.
    if focus_id == "confusion":
        return "Clarify understanding"

    # 3. Persona-level patterns with an unambiguous single objective.
    if persona in _PERSONA_CONVERSATION_GOALS:
        return _PERSONA_CONVERSATION_GOALS[persona]

    # 4. Career intent.
    if detected_intent == "career":
        return "Guide career decision"

    # 5. Success emotion.
    if detected_emotion == "success":
        return "Reinforce success"

    # 6. Positive, stable trajectory.
    if trajectory == "Positive Stability":
        return "Maintain momentum"

    # 7. Default -- no single signal dominates.
    return _DEFAULT_CONVERSATION_GOAL


# =========================================================
# COMPARISON QUESTION DETECTION (SPRINT D ADDITION)
# =========================================================
# Goal: academic answers like "difference between BFS and DFS" were
# being answered as a single-topic explainer (explain BFS, stop) with
# no actual comparison ever produced. This adds ONE new, deterministic,
# side-effect-free classifier -- same pattern as `determine_priority_
# focus` / `determine_conversation_goal` above -- that reads the raw
# question text and returns a plain bool: is this a comparison-style
# question or not?
#
# Deliberately NOT ML/embeddings-based, per the sprint requirement: a
# fixed list of regex patterns over the question text, case-insensitive,
# no external calls, no model, no randomness. This is pure text
# classification, easily unit-tested with plain strings in, bool out.
#
# This function does not change retrieval, the similarity gate,
# confidence, persona, or `detected_intent` -- it is consumed ONLY by
# mentor_service.py's `_build_output_format_instructions()` to decide
# which of two academic answer *structures* (comparison vs. normal) to
# instruct the model to use. `detected_intent == "academic"` still
# fully controls whether academic framing applies at all; this is a
# secondary, purely structural signal read from the same question text,
# exactly like `secondary_academic_signal` / `celebrated_academic_topic`
# in mentor_service.py are secondary reads of their own question text.
_COMPARISON_PATTERNS = [
    re.compile(r"\bdifference(s)?\s+(between|of|among)\b", re.IGNORECASE),
    re.compile(r"\bdiffer(s|ence)?\s+from\b", re.IGNORECASE),
    re.compile(r"\bcompar(e|ed|es|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdistinguish(es)?\s+between\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+is\s+better\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+one\s+should\s+i\s+use\b", re.IGNORECASE),
]


def detect_comparison_question(question: str) -> bool:
    """
    Deterministic, keyword/regex-based check for whether `question`
    is asking for a comparison between two or more things (e.g.
    "difference between BFS and DFS", "stack vs queue", "compare TCP
    and UDP", "binary tree vs BST"). No ML, no embeddings, no model
    call -- a plain string in, a plain bool out, so it is trivially
    unit-testable and cannot introduce any randomness or latency into
    the pipeline.

    Returns False for empty/None input. Matching is intentionally
    permissive (any single pattern hit counts) since false positives
    only cause a slightly more structured academic answer -- never a
    fabricated fact or a change to retrieval/evidence -- so the cost of
    an occasional over-match is low, while under-matching would
    reproduce exactly the reported bug (comparison asked, never given).
    """
    if not question:
        return False
    return any(pattern.search(question) for pattern in _COMPARISON_PATTERNS)


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


# =========================================================
# INTENT MODE (PHASE 1.2 ADDITION)
# =========================================================
# `detected_intent` was already computed by mentor_service.py
# (intent_classifier.classify_intent) and used only for retrieval
# routing. This adds the missing half: threading that same intent
# into the PROMPT so it also shapes the system ROLE and response
# style, not just which retrieval path runs.
#
# Persona and intent are deliberately kept independent:
#   - Persona (see _PSYCHOLOGICAL_FRAMEWORKS / _PERSONA_ACTIONS above)
#     controls HOW Aura speaks: tone, pacing, encouragement style.
#   - Intent (below) controls WHAT Aura discusses and which role it
#     plays for this specific turn: teacher, supportive mentor, career
#     advisor, encouragement coach, or balanced mentor.
# Priority Focus (determine_priority_focus) still decides which piece
# of evidence to anchor on, but it must never flatten or replace the
# persona-driven tone -- that constraint is stated explicitly in
# `_section_identity` below.
_INTENT_MODES = {
    "academic": {
        "role": "teacher",
        "style": (
            "lean on clear explanations and the retrieved evidence "
            "below; cite learning-material sources by name when "
            "discussing course content. "
            # SPRINT A ADDITION (Academic Answer Quality): reinforces,
            # earlier in the prompt, the same tutor-depth requirement
            # spelled out in mentor_service.py's academic-only output
            # format instructions -- no new data, no new section, just
            # a fuller description of what "clear explanations" means
            # for this intent.
            "Go beyond a one-line definition: explain how it works, "
            "cover key differences, steps, or characteristics as fits "
            "the topic, and add one example only if the retrieved "
            "material actually supports it."
        ),
    },
    "emotional": {
        "role": "supportive mentor",
        "style": (
            "stay reassuring and present; do not mention retrieval, "
            "citations, or missing evidence -- this is not a content "
            "question."
        ),
    },
    "career": {
        "role": "career advisor",
        "style": (
            "give a practical roadmap -- concrete next steps, "
            "projects, or internships -- over abstract theory."
        ),
    },
    "motivation": {
        "role": "encouragement coach",
        "style": (
            "focus on encouragement, habit formation, and building "
            "confidence through small, doable steps."
        ),
    },
    "general": {
        "role": "balanced mentor",
        "style": (
            "blend the persona's strategy with a balanced, "
            "conversational tone appropriate to whatever the student "
            "raises."
        ),
    },
}
_DEFAULT_INTENT_MODE = _INTENT_MODES["general"]


# =========================================================
# EMOTION MODE (PHASE 2.9 ADDITION)
# =========================================================
# `detected_emotion` is computed by the unified Message Analysis Module
# in intent_classifier.py (see that file's Phase 2.9 update) alongside
# `detected_intent`, from the SAME question text. It is threaded here,
# into the Identity section, so it shapes ONLY how Aura's reply
# emotionally opens and supports the learner -- never what evidence is
# retrieved, never confidence, never recommendation selection.
#
# Persona, intent, and emotion remain three independent dimensions:
#   - Persona (above) = HOW Aura speaks (tone, pacing, encouragement
#     style), derived from the learner's behavioural archetype.
#   - Intent (above)  = WHAT Aura discusses and which role it plays.
#   - Emotion (below) = HOW Aura should emotionally frame THIS reply's
#     opening and support, derived from the current message alone.
# None of these three overrides another; Priority Focus still decides
# which evidence to anchor on regardless of detected emotion.
_EMOTION_MODES = {
    "stress": {
        "framing": (
            "open by acknowledging the pressure they're under before "
            "anything else; keep the reply calm and unhurried."
        ),
    },
    "frustration": {
        "framing": (
            "open by validating that this is genuinely frustrating; "
            "avoid sounding dismissive or rushing straight to a fix."
        ),
    },
    "confusion": {
        "framing": (
            "open by normalizing that this is a confusing point; keep "
            "the explanation patient and step-by-step."
        ),
    },
    "anxiety": {
        "framing": (
            "open with a steadying, reassuring tone; avoid language "
            "that could heighten worry, and keep pacing gentle."
        ),
    },
    "hopelessness": {
        "framing": (
            "open with genuine warmth and reassurance that things are "
            "workable; never minimize the feeling, and avoid piling on "
            "tasks or pressure."
        ),
    },
    "burnout": {
        "framing": (
            "open by acknowledging how drained they sound; favor rest "
            "and small, low-effort next steps over ambitious asks."
        ),
    },
    "success": {
        "framing": (
            "open by genuinely celebrating this win before moving on "
            "to anything else."
        ),
    },
    "confidence": {
        "framing": (
            "open by affirming their confidence and momentum; keep the "
            "tone encouraging rather than cautionary."
        ),
    },
    "neutral": {
        "framing": (
            "open in a natural, even tone -- no particular emotional "
            "framing is called for here."
        ),
    },
}
_DEFAULT_EMOTION_MODE = _EMOTION_MODES["neutral"]


def get_emotion_mode(detected_emotion: str) -> Dict[str, str]:
    """
    Public helper (mirrors get_intent_mode) returning the {framing}
    fragment for a detected_emotion, or the "neutral" default for any
    unrecognized value. Pure lookup -- no new classification logic;
    emotion itself is still produced exclusively by the unified Message
    Analysis Module (intent_classifier.classify_emotion /
    analyze_message).
    """
    return _EMOTION_MODES.get(detected_emotion, _DEFAULT_EMOTION_MODE)


# =========================================================
# EMOTION INTENSITY MODE (PHASE 3.1 ADDITION)
# =========================================================
# `detected_emotion_intensity` is computed by the unified Message
# Analysis Module (intent_classifier.classify_emotion_intensity /
# analyze_message), from the SAME question text as emotion, and is a
# FOURTH independent dimension threaded into the Identity section
# alongside persona, intent, and emotion. It controls ONLY pacing,
# validation depth, and how quickly the reply moves from emotional
# support into guidance -- it has no path into retrieval, confidence,
# evidence prioritization, SHAP reasoning, persona, or recommendation
# selection (those are computed entirely elsewhere, upstream of prompt
# construction, and never read this value).
#
#   - "High"   -- the emotion is expressed in absolute/overwhelming
#     terms. Aura should slow down, validate and reassure heavily
#     BEFORE any advice, and keep the turn's guidance minimal (fewer
#     suggested actions) so the reply doesn't pile tasks onto someone
#     already overwhelmed.
#   - "Medium" -- an ordinary, amplified-but-manageable statement of
#     the emotion. Balanced reassurance, normal pacing, guidance
#     follows naturally after a brief acknowledgement.
#   - "Low"    -- a mild or hedged statement of the emotion (or no
#     emotion detected at all). A brief acknowledgement is enough;
#     Aura should move into guidance quickly rather than dwelling on
#     validation.
#
# NOTE: "fewer recommendations" for High intensity is a PROMPT
# INSTRUCTION to the LLM about how much guidance to surface in its own
# reply text -- it does not change `recommendation_engine.py`,
# `determine_priority_focus`, or which student/educator recommendation
# was selected upstream. Those remain exactly as computed before this
# phase; only how much of that guidance the LLM chooses to elaborate on
# in its conversational reply is nudged by this instruction.
_EMOTION_INTENSITY_MODES = {
    "High": {
        "pacing": (
            "slow your pacing right down; lead with strong, unhurried "
            "validation and reassurance BEFORE offering any advice; "
            "keep guidance to at most one small, gentle next step "
            "rather than a list of recommendations."
        ),
    },
    "Medium": {
        "pacing": (
            "offer balanced reassurance at a normal pace, then move "
            "naturally into guidance."
        ),
    },
    "Low": {
        "pacing": (
            "a brief acknowledgement is enough here; transition "
            "quickly into the actual guidance."
        ),
    },
}
_DEFAULT_EMOTION_INTENSITY_MODE = _EMOTION_INTENSITY_MODES["Medium"]


def get_emotion_intensity_mode(detected_emotion_intensity: str) -> Dict[str, str]:
    """
    Public helper (mirrors get_intent_mode / get_emotion_mode) returning
    the {pacing} fragment for a detected_emotion_intensity, or the
    "Medium" default for any unrecognized value. Pure lookup -- no new
    classification logic; intensity itself is still produced exclusively
    by the unified Message Analysis Module
    (intent_classifier.classify_emotion_intensity / analyze_message).
    """
    return _EMOTION_INTENSITY_MODES.get(
        detected_emotion_intensity, _DEFAULT_EMOTION_INTENSITY_MODE
    )


# =========================================================
# EMOTION TRAJECTORY MODE (PHASE 3.2 ADDITION)
# =========================================================
# `detected_emotion_trajectory` is computed by
# intent_classifier.classify_emotional_trajectory() /
# analyze_message() from `context.chat_history` (read-only, no new
# storage) plus the current turn's emotion. It is threaded here as a
# FIFTH independent Identity dimension, alongside persona, intent,
# emotion, and emotion intensity. It controls ONLY continuity-aware
# emotional framing across turns -- e.g. not repeating the same
# reassurance every message when distress is holding steady, or
# naturally acknowledging an improving trend -- and has no path into
# retrieval, confidence, evidence prioritization, SHAP reasoning,
# persona, or recommendation selection.
#
# Keys below match `intent_classifier.VALID_EMOTION_TRAJECTORIES`
# exactly. The lookup in `get_emotion_trajectory_mode` is
# case-insensitive so the literal default sentinel `"neutral"` (lower-
# case, matching the Phase 3.2 spec's default value for
# `build_system_prompt`) resolves to the same framing as the
# classifier's own `"Neutral"` output, without needing two separate
# entries for what is conceptually one state.
_EMOTION_TRAJECTORY_MODES = {
    "stable emotional distress": {
        "framing": (
            "this student's distress has remained steady across "
            "recent turns -- maintain continuity without repeatedly "
            "restating the same reassurance."
        ),
    },
    "escalating distress": {
        "framing": (
            "this student's emotional distress appears to be "
            "escalating across recent turns -- respond with "
            "heightened care and avoid moving too quickly into "
            "task-focused advice."
        ),
    },
    "improving emotional state": {
        "framing": (
            "this student's emotional state appears to be improving "
            "across recent turns -- reflect that shift naturally "
            "rather than repeating earlier concern."
        ),
    },
    "positive stability": {
        "framing": (
            "this student has maintained a positive, confident tone "
            "across recent turns -- keep the encouraging momentum "
            "going."
        ),
    },
    "neutral": {
        "framing": (
            "no notable emotional trend across recent turns -- no "
            "special continuity framing is needed here."
        ),
    },
    "mixed emotional pattern": {
        "framing": (
            "this student's emotional signals across recent turns are "
            "mixed -- respond to what they've expressed in this "
            "specific message rather than assuming a consistent trend."
        ),
    },
}
_DEFAULT_EMOTION_TRAJECTORY_MODE = _EMOTION_TRAJECTORY_MODES["neutral"]


def get_emotion_trajectory_mode(detected_emotion_trajectory: str) -> Dict[str, str]:
    """
    Public helper (mirrors get_intent_mode / get_emotion_mode /
    get_emotion_intensity_mode) returning the {framing} fragment for a
    detected_emotion_trajectory, or the "neutral" default for any
    unrecognized value. Lookup is case-insensitive (see module note
    above) since the default parameter value is the lowercase sentinel
    "neutral" while the classifier itself returns "Neutral". Pure
    lookup -- no new classification logic; trajectory itself is still
    produced exclusively by the unified Message Analysis Module
    (intent_classifier.classify_emotional_trajectory / analyze_message).
    """
    key = (detected_emotion_trajectory or "").strip().lower()
    return _EMOTION_TRAJECTORY_MODES.get(key, _DEFAULT_EMOTION_TRAJECTORY_MODE)


def get_intent_mode(detected_intent: str) -> Dict[str, str]:
    """
    Public helper (mirrors get_persona_actions/get_psychological_framework)
    returning the {role, style} fragment for a detected_intent, or the
    "general" default for any unrecognized value. Pure lookup -- no new
    classification logic; intent itself is still produced exclusively
    by intent_classifier.classify_intent().
    """
    return _INTENT_MODES.get(detected_intent, _DEFAULT_INTENT_MODE)


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


def _section_identity(
    context: StudentContext,
    detected_intent: str,
    detected_emotion: str = "neutral",
    detected_emotion_intensity: str = "Medium",
    detected_emotion_trajectory: str = "neutral",
) -> str:
    """
    PHASE 1.2 ADDITION -- promotes Persona from "one small context
    section" to part of Aura's system identity, and threads
    `detected_intent` (already computed by mentor_service.py, until
    now used only for retrieval routing) into the same section so it
    shapes the system ROLE and response style.

    Reuses `_section_persona` verbatim for the persona fragment (same
    persona/strategy/lead-action text as before -- no persona logic
    duplicated or changed) and `get_intent_mode` for the intent
    fragment. The only new content is the framing sentence that
    establishes Aura is mentoring THIS learner right now, plus the
    explicit independence rule: persona governs HOW Aura speaks,
    intent governs WHAT it focuses on, and neither may suppress the
    other -- including Priority Focus below, which decides which
    evidence to anchor on but must never override persona-driven tone.

    PHASE 2.9 UPDATE (Emotion Response Layer): adds a third,
    independent fragment from `get_emotion_mode(detected_emotion)`, the
    same way intent was added in Phase 1.2. `detected_emotion` defaults
    to "neutral" so any existing caller of `_section_identity` (or of
    `build_system_prompt`, below) that doesn't pass it keeps getting
    the original persona+intent framing plus a no-op neutral emotion
    line -- no existing behaviour changes for a caller that hasn't
    adopted the new parameter. Persona/intent/emotion are stated here
    as three explicitly independent dimensions: persona defines the
    mentoring style, intent defines the discussion focus, and emotion
    defines only how the reply should begin and emotionally support the
    learner -- it does not change what evidence is retrieved, what
    intent routing occurred, or what the Priority Focus below anchors
    on.

    PHASE 3.1 UPDATE (Emotion Intensity Layer): adds a FOURTH,
    independent fragment from
    `get_emotion_intensity_mode(detected_emotion_intensity)`.
    `detected_emotion_intensity` defaults to "Medium" so any existing
    caller that doesn't pass it keeps getting the original
    persona+intent+emotion framing plus a balanced, normal-pacing
    intensity line -- no existing behaviour changes for a caller that
    hasn't adopted the new parameter. Intensity controls ONLY how
    strongly Aura validates, reassures, and paces the reply (and how
    much guidance it front-loads) -- it does not change what evidence
    is retrieved, which intent/emotion was detected, which persona is
    active, or what the Priority Focus below anchors on.

    PHASE 3.2 UPDATE (Emotional Trajectory Memory): adds a FIFTH,
    independent fragment from
    `get_emotion_trajectory_mode(detected_emotion_trajectory)`, one
    short continuity instruction derived from
    `intent_classifier.classify_emotional_trajectory()` (read-only over
    `context.chat_history` -- no new storage). `detected_emotion_trajectory`
    defaults to "neutral" so any existing caller that doesn't pass it
    keeps getting the original persona+intent+emotion+intensity framing
    plus a no-op "no notable trend" line -- no existing behaviour
    changes for a caller that hasn't adopted the new parameter.
    Trajectory controls ONLY continuity of emotional framing ACROSS
    turns (e.g. not repeating the same reassurance every message when
    distress is holding steady) -- it does not change what evidence is
    retrieved, which intent/emotion/intensity was detected, which
    persona is active, or what the Priority Focus below anchors on.
    All five dimensions are named explicitly in the independence
    sentence below so the model cannot conflate "how has this trended
    across turns" with "what to discuss," "which evidence to
    prioritize," or any of the other four.
    """
    persona_fragment = _section_persona(context)
    mode = get_intent_mode(detected_intent)
    emotion_mode = get_emotion_mode(detected_emotion)
    intensity_mode = get_emotion_intensity_mode(detected_emotion_intensity)
    trajectory_mode = get_emotion_trajectory_mode(detected_emotion_trajectory)
    return (
        f"IDENTITY: You are Aura, mentoring {context.student_name} in "
        f"this conversation. {persona_fragment} Detected intent for "
        f"this turn: {detected_intent} -- act as a {mode['role']}: "
        f"{mode['style']} Detected emotion for this turn: "
        f"{detected_emotion} -- {emotion_mode['framing']} Detected "
        f"emotion intensity for this turn: {detected_emotion_intensity} "
        f"-- {intensity_mode['pacing']} Emotional trajectory across "
        f"recent turns: {detected_emotion_trajectory} -- "
        f"{trajectory_mode['framing']} Persona, intent, emotion, "
        f"emotion intensity, and emotion trajectory are independent: "
        f"persona controls the mentoring style (tone, pacing, "
        f"encouragement); intent controls the conversation purpose "
        f"(what you discuss and how you structure the reply); emotion "
        f"controls the emotional framing (how the reply should begin "
        f"and emotionally support the learner); emotion intensity "
        f"controls how strongly Aura should validate, reassure, and "
        f"pace the response; emotion trajectory controls only "
        f"continuity of that emotional framing across recent turns. "
        f"None of these five may suppress or override the others -- "
        f"including the PRIORITY FOCUS below, which decides WHAT "
        f"evidence to anchor on but must never flatten or override the "
        f"persona-, emotion-, intensity-, or trajectory-driven framing "
        f"established here."
    )


def _section_conversation_goal(conversation_goal: str = _DEFAULT_CONVERSATION_GOAL) -> str:
    """
    PHASE 3.3 ADDITION (Adaptive Conversation Planning). One short,
    additional section placed immediately after Identity. States the
    already-computed `conversation_goal` (see `determine_conversation_goal`
    above) as this turn's mentoring objective -- it does not repeat or
    contradict anything in the Identity section (persona/intent/
    emotion/intensity/trajectory framing there is untouched), and it
    does not itself pick or describe a recommendation (Priority Focus,
    further down, still owns that).

    `conversation_goal` defaults to `_DEFAULT_CONVERSATION_GOAL`
    ("General Support") so any existing caller of `build_system_prompt`
    that doesn't pass one keeps getting a harmless, generic objective
    line -- no existing behaviour changes for a caller that hasn't
    adopted the new parameter.

    Kept deliberately short (one sentence), per the Phase 3.3 spec.
    """
    goal_text = (conversation_goal or _DEFAULT_CONVERSATION_GOAL).strip()
    lowered = goal_text.lower()
    return (
        "CONVERSATION GOAL: Today's mentoring objective is to "
        f"{lowered} before suggesting any academic action."
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


def select_chunks_for_prompt(retrieved_chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    SPRINT 8 ADDITION. Returns exactly the chunks (and only those) that
    _section_learning_material will render into the prompt. Factored
    out into a public function so mentor_service.py can build its
    "Sources:" citation list from the SAME slice the model actually
    saw, rather than re-deriving (and risking disagreement with) the
    cap applied here. Pure slicing -- no ranking, filtering, or scoring
    logic is added; retrieved_chunks is already ranked by
    retriever.retrieve() (untouched).
    """
    return retrieved_chunks[:_MAX_CHUNKS_IN_PROMPT]


def _format_retrieved_chunk_line(item: Dict[str, str]) -> str:
    """
    SPRINT 8 ADDITION. Renders one retrieved chunk with its citation
    metadata inline, so the LLM can (a) ground its answer in the
    excerpt text and (b) name the correct source when citing evidence,
    using ONLY source names it was actually shown here -- never a
    source it invents. source, chunk_id, and similarity_score are
    all fields retrieval.py already attaches to each chunk; nothing is
    computed here beyond formatting and the existing per-chunk
    character cap.
    """
    chunk_text = item.get("chunk", "")[:_MAX_CHARS_PER_CHUNK]
    source = item.get("source") or item.get("topic") or "unknown source"
    chunk_id = item.get("chunk_id", "unknown")
    similarity = item.get("similarity_score")
    similarity_str = f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "n/a"
    return f"[source: {source} | id: {chunk_id} | similarity: {similarity_str}] {chunk_text}"


def _section_learning_material(retrieved_chunks: List[Dict[str, str]]) -> str:
    """
    goal #4: use only the most relevant retrieved chunks instead of
    concatenating all of them. retrieved_chunks is already ranked by
    relevance by retriever.retrieve() (untouched, unchanged) -- this
    function only limits how many of those already-ranked chunks, and
    how much of each, get embedded into the prompt text.

    SPRINT 8 UPDATE (Evidence Citations / Retrieval Safety): each
    rendered chunk now carries its source name, chunk id, and
    similarity score inline (see _format_retrieved_chunk_line), and
    the section explicitly instructs the model to cite ONLY the exact
    source names shown here, and to say so plainly rather than answer
    from outside knowledge if these excerpts don't cover the question.
    The chunk selection itself (select_chunks_for_prompt) and the
    existing character caps are unchanged from before.
    """
    if not retrieved_chunks:
        return (
            "Learning material: none retrieved for this query. Do not "
            "answer content questions from outside knowledge -- say "
            "plainly that no matching material was found."
        )

    top_chunks = select_chunks_for_prompt(retrieved_chunks)
    lines = [_format_retrieved_chunk_line(item) for item in top_chunks]
    combined = " ".join(lines)[:_MAX_TOTAL_MATERIAL_CHARS]
    return (
        "Learning material (cite ONLY these exact source names when "
        "referencing course content; if they do not contain the "
        "answer, say so instead of using outside knowledge): "
        f"{combined}"
    )


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


def _section_hallucination_safety() -> str:
    """
    SPRINT 8 ADDITION (Hallucination Safety Hardening). A dedicated,
    high-salience section placed right after the raw evidence sections
    (Learning Material included) and before the final Response
    Instructions, restating -- as an explicit, mandatory rule rather
    than an implicit expectation -- the core hallucination-safety
    contract: never fabricate, never answer content questions from
    outside the retrieved evidence, and say so plainly when evidence is
    thin instead of guessing.

    This is a defense-in-depth backstop, not the primary safety
    mechanism: mentor_service.py's SIMILARITY_THRESHOLD gate already
    prevents the LLM from being called at all when the retrieved
    material is clearly too weak to answer from (Req: Similarity
    Threshold). This section covers the remaining case where the LLM
    IS called (evidence cleared the threshold) but the student's
    specific question may still reach beyond what the retrieved
    excerpts actually cover.
    """
    return (
        "HALLUCINATION SAFETY (mandatory, overrides stylistic "
        "instructions if they ever conflict): Never invent facts, "
        "statistics, quotes, or sources. For any question about course "
        "content, answer ONLY from the Learning material excerpts "
        "above and cite them by their exact source name; never invent "
        "a source name that was not shown to you. If the excerpts do "
        "not contain the answer, say so plainly (e.g. 'I don't know' "
        "or 'Insufficient evidence was retrieved to answer this "
        "question reliably.') instead of guessing. General "
        "conversation, motivation, or study-skills questions that do "
        "not depend on specific course content are not subject to this "
        "citation requirement, but must still never state a specific "
        "fact about this student (risk, persona, cognitive state) that "
        "was not explicitly provided above."
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
    # PHASE 1.2 UPDATE: restructured from a flat instruction block into
    # an explicitly prioritized 1-7 list, per the Phase 1.2 review. No
    # constraint from the previous version was dropped -- each of the
    # 7 numbered items below still covers the same ground (tone,
    # intent, evidence use, generic-advice ban, repetition ban,
    # internals/hallucination safety, output format); they are simply
    # ordered so the model applies them in the priority order a small
    # instruction-follower benefits from most: identity/tone first,
    # then intent, then substance, then safety/format last.
    return (
        "Instructions (apply in this priority order): "
        "1. Maintain the persona-specific tone, pacing, and "
        "encouragement style set out in IDENTITY above throughout the "
        "reply. "
        "2. Respect the detected intent's role from IDENTITY — teach "
        "with evidence and citations for academic, stay reassuring "
        "with no talk of retrieval for emotional, give a practical "
        "roadmap for career, build confidence and habits for "
        "motivation, or stay a balanced mentor for general. "
        "3. Use the evidence above naturally — anchor your reply and "
        "both recommendations on the ONE signal named in PRIORITY "
        "FOCUS; treat EVIDENCE FUSION's secondary/supporting items as "
        "background only, and lower-priority items only if directly "
        "asked; follow the CONVERSATION PLAN's opening goal, teaching "
        "style, and recommendation style rather than inventing your "
        "own structure. "
        "4. Avoid generic recommendations (e.g. 'take deep breaths', "
        "'study more', 'stay positive') unless tied to the priority "
        "focus and a concrete evidence point; both recommendations "
        "must follow directly from what your reply just said, not "
        "introduce new, unrelated advice. "
        "5. Avoid repeating previous recommendations or openings — "
        "respect CONTINUITY: do not reopen with the same phrasing as "
        "last time, do not repeat a previously-used recommendation, "
        "and vary your opening phrase rather than defaulting to 'I "
        "understand' or 'It sounds like' every time. "
        "6. Never expose internal implementation — never reveal raw "
        "scores, thresholds, or pipeline internals unless asked, never "
        "invent unavailable data, and per HALLUCINATION SAFETY above: "
        "never fabricate a fact, statistic, or source, and never "
        "answer a course-content question beyond what the Learning "
        "material excerpts support. "
        "7. Follow the output format — you MUST output all three "
        "section markers below, exactly as given, every time; do not "
        "omit or rename them."
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
    "Identity (Persona + Intent + Emotion + Intensity + Trajectory)",  # Phase 1.2 (was "Persona"); Phase 2.9 adds Emotion; Phase 3.1 adds Intensity; Phase 3.2 adds Trajectory
    "Conversation Goal",  # Phase 3.3 (Adaptive Conversation Planning)
    "Priority Focus",
    "Academic Risk",
    "SHAP",
    "Cognitive",
    "Evidence Fusion",       # Sprint 7
    "Conversation Plan",     # Sprint 7
    "Student Profile",
    "History",
    "Continuity",            # Sprint 7
    "Confidence Weighting",  # Sprint 7
    "Learning Material",
    "Hallucination Safety",  # Sprint 8
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
    detected_intent: str = "general",
    detected_emotion: str = "neutral",
    detected_emotion_intensity: str = "Medium",
    detected_emotion_trajectory: str = "neutral",
    conversation_goal: str = _DEFAULT_CONVERSATION_GOAL,
) -> str:
    """
    Assembles the full structured system prompt from individual,
    independently-testable sections.

    PHASE 3.3 UPDATE (Adaptive Conversation Planning): new
    `conversation_goal` parameter, defaulting to `_DEFAULT_CONVERSATION_GOAL`
    ("General Support") so any existing caller that doesn't pass it
    keeps working exactly as before (backward compatible, same pattern
    as every prior phase's additive parameter). mentor_service.py
    computes this value via `determine_conversation_goal()` (see above)
    and passes it in; it is rendered as ONE new, short section
    (`_section_conversation_goal`) placed immediately after Identity --
    no other section, no retrieval/confidence/reasoning/recommendation
    logic, and no existing section's internal computation or ordering
    relative to each other was changed. This value is never exposed
    through the API response.

    PHASE 3.2 UPDATE (Emotional Trajectory Memory): new
    `detected_emotion_trajectory` parameter, defaulting to "neutral" so
    any existing caller that doesn't pass it keeps working exactly as
    before (backward compatible, same pattern as
    `detected_emotion_intensity`'s Phase 3.1 addition). It is threaded
    only into `_section_identity`, alongside persona, intent, emotion,
    and emotion intensity -- no other section, no
    retrieval/confidence/reasoning/recommendation logic, and no section
    ordering changed in this update.

    PHASE 3.1 UPDATE (Emotion Intensity Layer): new
    `detected_emotion_intensity` parameter, defaulting to "Medium" so
    any existing caller that doesn't pass it keeps working exactly as
    before (backward compatible, same pattern as `detected_emotion`'s
    Phase 2.9 addition and `detected_intent`'s Phase 1.2 addition). It
    is threaded only into `_section_identity`, alongside persona,
    intent, and emotion -- no other section, no
    retrieval/confidence/reasoning/recommendation logic, and no
    section ordering changed in this update.

    PHASE 2.9 UPDATE (Emotion Response Layer): new `detected_emotion`
    parameter, defaulting to "neutral" so any existing caller that
    doesn't pass it keeps working exactly as before (backward
    compatible, same as `detected_intent`'s Phase 1.2 addition). It is
    threaded only into `_section_identity`, alongside persona and
    intent -- no other section, no retrieval/confidence/reasoning
    logic, and no section ordering changed in this update.

    PHASE 1.2 UPDATE (persona + intent driven prompt): two changes to
    this function, both purely about WHICH sections run and in WHAT
    ORDER -- no section's internal computation, no schema, and no
    other pipeline (retrieval/confidence/reasoning) was touched:

      1. New `detected_intent` parameter (defaults to "general" so any
         existing caller that doesn't pass it keeps working exactly as
         before). mentor_service.py already computes this value via
         intent_classifier.classify_intent() for retrieval routing;
         it is now also threaded here so intent shapes the system
         ROLE and response style, not just which retrieval path runs.
         `_section_persona` was replaced in the section list below by
         `_section_identity(context, detected_intent)`, which reuses
         `_section_persona` internally (identical persona text) and
         adds the intent-driven role/style framing next to it.

      2. Section order now follows: SYSTEM ROLE -> IDENTITY (persona +
         intent) -> Priority Focus -> Evidence (Academic Risk, SHAP,
         Cognitive, Evidence Fusion) -> Conversation Plan -> Student
         Context (Student Profile, History, Continuity, Confidence
         Weighting) -> Learning Material -> Hallucination Safety ->
         Instructions. This puts identity (WHO the model is speaking
         to and acting as) before any evidence, so the model reads
         persona/intent framing before it starts weighing facts. Every
         individual section function is unchanged; only their position
         in this list moved.

    `determine_priority_focus(context)` is still called ONCE here and
    shared with `_section_priority_focus`, `_section_evidence_fusion`,
    and `_section_conversation_plan`, exactly as before.
    """

    focus = determine_priority_focus(context)

    sections = [
        _section_system_role(),
        _section_identity(
            context,
            detected_intent,
            detected_emotion,
            detected_emotion_intensity,
            detected_emotion_trajectory,
        ),
        _section_conversation_goal(conversation_goal),
        _section_priority_focus(focus),
        _section_academic_risk(context),
        _section_shap_summary(context),
        _section_cognitive_intelligence(context),
        _section_evidence_fusion(context, focus),
        _section_conversation_plan(context, focus),
        _section_student_profile(context),
        _section_conversation_history(context.chat_history),
        _section_continuity(context),
        _section_confidence_weighting(context),
        _section_learning_material(retrieved_chunks),
        _section_hallucination_safety(),
        _section_response_instructions(),
    ]

    # TEMPORARY DEBUGGING: report each section's size before returning.
    # No section text is modified here — this only reads len() on the
    # exact same strings that get joined below.
    _debug_print_section_sizes(list(zip(_DEBUG_SECTION_LABELS, sections)))

    return "\n\n".join(sections)
