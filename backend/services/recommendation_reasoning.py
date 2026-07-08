"""
recommendation_reasoning.py

PHASE 2.3 ADDITION (Dynamic Personalized Recommendation Generation).
PHASE 2.4 UPDATE (Human-Mentor Reasoning Quality).

SCOPE / HARD CONSTRAINTS FOR THIS PHASE
-----------------------------------------
- `recommendation_engine.py` (Phase 2.2) is UNCHANGED. It still decides
  WHICH intervention is selected — persona-aware, risk-aware,
  intent-aware, rotation-aware. This module never picks or overrides
  an action; it only explains it.
- No API, FastAPI, frontend, or database changes. This module is a
  pure function library, imported only by mentor_service.py's
  recommendation-rendering path.
- Nothing here is exposed directly to API consumers. The rationale
  this module builds is consumed ONLY by the (Phase 2.4-updated)
  renderers in mentor_service.py to shape the final
  student_recommendation / educator_recommendation strings — it is
  never returned as its own field.
- Never surfaces a raw score, SHAP value, or feature name. Every
  natural-language clause below is a fixed, hand-written phrase chosen
  from a small set of already-computed CATEGORICAL signals (a
  threshold crossed or not, a direction, a boolean) — the same
  deterministic, no-fabrication discipline the rest of this codebase
  follows (context_builder.py, confidence.py, reasoning_layer.py). If
  a signal doesn't clearly map to one of the phrases below, it is
  simply omitted rather than guessed at.
- Still fully deterministic: no randomness, no LLM calls. Same intent
  + same evidence always yields the same rationale.

PHASE 2.4 CHANGE SUMMARY (see mentor_service.py's own Phase 2.4 note
for the renderer-side half of this change)
-----------------------------------------
The Phase 2.3 version built `student_opener` + `question_snippet` to
echo the student's own words back to them ("You mentioned..."), and
built `situation_phrase` by comma-joining independently-produced
fragments (human_reason + cognitive clause + SHAP clause). Both read
as template filling rather than a mentor reasoning through a
situation. Phase 2.4 replaces both:

1. `opening_observation` replaces `student_opener` +
   `question_snippet`. Instead of echoing the question, it infers what
   kind of thing is going on from `detected_intent` — a fixed,
   hand-written observation per intent category (still no randomness,
   still no LLM, still nothing invented about the specific question's
   content, since the phrase never varies by anything BUT the already-
   classified intent).
2. `situation_student` / `situation_educator` replace `situation_phrase`.
   Instead of joining fragments with "alongside", cognitive and SHAP
   observations are now written as full independent clauses (with a
   real subject: "you"/"the learner") and folded into ONE sentence
   with "and" — the same information, composed as prose a person would
   actually write, not a list.
3. `risk_sentence_student` / `risk_sentence_educator` are new: a fixed,
   hand-written sentence per risk level (High/Medium/Low) that
   describes what the risk level MEANS in plain language, never a
   number. Still gated the same way the old code avoided repeating
   "elevated risk" when the focus itself already implies it (e.g.
   burnout already means High risk by construction).
4. `memory_clause_student` / `memory_clause_educator` now reference the
   most recently used real action tag (from
   `derive_persona_memory()`'s `used_action_tags` — unchanged data
   source) turned into plain words, instead of a single generic
   sentence reused for every learner with any history. If there is no
   real history, both are still `None` — nothing is invented.
5. `closing_line` is new: a fixed, hand-written encouraging sentence
   per risk level, giving renderers a natural place to end the
   student-facing recommendation on a supportive note (item #7 of the
   Phase 2.4 brief) without inventing anything learner-specific.

Every one of the above is still built from data
`_fallback_recommendations()` in mentor_service.py already has on hand
(persona's `human_reason`, detected_intent, risk_level, focus_id,
cognitive_state, shap_explanation, question, memory) — nothing new is
fetched, computed elsewhere, or asked of an LLM.

PHASE 2.5 UPDATE (Dynamic Mentor Personality)
-----------------------------------------
Adds one new, optional parameter to `build_recommendation_rationale()`
— `persona` — and a small persona-voice lookup table (`_PERSONA_VOICE`)
used ONLY to select a fixed opening sentence, closing sentence,
encouragement phrasing template, and a cap on situation-sentence
length, for the STUDENT-facing recommendation. This is a HOW-level
addition on top of the existing WHAT-level content (situation, risk
framing) built above — it does not change `human_reason`, does not
change which cognitive/SHAP signals are eligible, does not touch
`determine_priority_focus()` or `recommendation_engine.py`'s
selection, and does not affect the educator recommendation's
professional case-note register (Phase 2.4's decision). See
`_PERSONA_VOICE`'s own comment block below for the full rationale.

"""

from typing import Any, Dict, List, Optional


# =========================================================
# SPRINT 3 — DETERMINISTIC WORDING VARIATION (SURFACE ONLY)
# =========================================================
# Several lookups below (`_OBSERVATION_BY_INTENT`, `_RISK_SENTENCE_*`,
# `_PERSONA_VOICE`, `_PERSONA_COMMUNICATION_PROFILE`) previously mapped
# a coarse category (intent / risk level / persona) to exactly ONE
# fixed string. Two different conversations landing on the same
# category produced word-for-word identical sentences even though the
# learner, question, or history differed. This adds small (2-variant)
# pools to those same entries and a tiny deterministic selector below
# that picks one variant using signals ALREADY computed elsewhere in
# this pipeline: persona, focus_id, risk_level, detected_intent, and
# how many actions this learner has already been given
# (`len(used_action_tags)`, from the unchanged `derive_persona_memory`
# output). No randomness, no new data source, no LLM call: identical
# signals always resolve to the identical variant. This never adds a
# slot, never invents a phrase, and never touches
# `determine_priority_focus()` or
# `recommendation_engine.select_recommendation()`.

def _stable_index(modulo: int, *signal_parts: Any) -> int:
    """Deterministic (never random) index in range [0, modulo)."""
    if modulo <= 1:
        return 0
    key = "|".join("" if part is None else str(part) for part in signal_parts)
    checksum = sum(ord(ch) for ch in key)
    return checksum % modulo


def _select_variant(pool: Any, salt: str, *signal_parts: Any) -> Optional[str]:
    """
    Deterministically picks one string from `pool` (a list of 2-3
    hand-written variants). `salt` distinguishes this slot from other
    slots that share the same signal values, so e.g. `persona_opening`
    and `persona_closing` don't always land on the same index. `pool`
    may also be `None` or a bare string (entries not converted to a
    pool, e.g. the default communication profile's `None` phrases) --
    both pass through unchanged, so this is safe to call everywhere.
    """
    if pool is None or isinstance(pool, str):
        return pool
    if not pool:
        return None
    return pool[_stable_index(len(pool), salt, *signal_parts)]


# =========================================================
# COGNITIVE -> NATURAL CLAUSES (never raw numbers)
# =========================================================
# Thresholds intentionally reuse the SAME cutoffs prompt_builder.py's
# determine_priority_focus() already uses (_CONFUSION_THRESHOLD=70,
# _ENGAGEMENT_THRESHOLD=50) so a clause is only added here when the
# underlying signal is ALSO one the rest of the pipeline treats as
# meaningful — this module doesn't invent new sensitivity, it just
# narrates the same evidence in prose.
#
# PHASE 2.4: each clause is now a full independent clause (own
# subject), voiced for either the student ("you") or the educator
# ("the learner"), so it can be folded into one flowing sentence
# instead of being comma-joined as a bare noun phrase.

_CONFUSION_THRESHOLD = 70
_ENGAGEMENT_THRESHOLD = 50


def _describe_cognitive_state(
    cognitive: Optional[Dict[str, Any]],
    voice: str,
) -> List[str]:
    if not cognitive:
        return []

    clauses: List[str] = []
    confusion = cognitive.get("confusion_score")
    engagement = cognitive.get("engagement_score")

    if confusion is not None and confusion >= _CONFUSION_THRESHOLD:
        clauses.append(
            "you've been having some real difficulty following the "
            "material clearly"
            if voice == "student"
            else "the learner has shown some real difficulty following "
            "the material clearly"
        )

    if engagement is not None and engagement < _ENGAGEMENT_THRESHOLD:
        clauses.append(
            "your engagement has gradually declined over time"
            if voice == "student"
            else "the learner's engagement has gradually declined over time"
        )

    return clauses


# =========================================================
# SHAP -> NATURAL CLAUSES (never feature names or magnitudes)
# =========================================================
# PHASE 2.4: previously only the single top-ranked feature was
# considered. Now every top feature with a positive impact direction
# is checked against the same small set of known, easily-explained
# categories (still hand-written, still no invented category), so more
# than one behavioural signal can be woven into the situation sentence
# when the backend actually produced more than one. An unrecognized
# feature name is still simply skipped rather than guessed at.

_SHAP_CATEGORY_CLAUSES = {
    "inactiv": {
        "student": "there's been a recent stretch of inactivity",
        "educator": "a recent stretch of inactivity stands out",
    },
    "engagement": {
        "student": "your engagement with the course material has recently dropped",
        "educator": "engagement with course material has recently dropped",
    },
    "assessment": {
        "student": "keeping up a consistent study routine has been difficult recently",
        "educator": "maintaining a consistent study routine has recently been difficult",
    },
    "score": {
        "student": "keeping up a consistent study routine has been difficult recently",
        "educator": "maintaining a consistent study routine has recently been difficult",
    },
    "consistency": {
        "student": "keeping up a consistent study routine has been difficult recently",
        "educator": "maintaining a consistent study routine has recently been difficult",
    },
}


def _describe_shap(
    shap_explanation: Optional[Dict[str, Any]],
    voice: str,
) -> List[str]:
    if not shap_explanation:
        return []

    top_features = shap_explanation.get("top_features") or []
    if not top_features:
        return []

    clauses: List[str] = []
    seen_categories = set()

    for top in top_features:
        if top.get("impact_direction") != "positive":
            continue

        feature = str(top.get("feature", "")).lower()
        for category, phrasing in _SHAP_CATEGORY_CLAUSES.items():
            if category in seen_categories:
                continue
            if category in feature:
                clauses.append(phrasing[voice])
                seen_categories.add(category)
                break

    return clauses


# PHASE 2.4 ADDITION: cognitive_state's "engagement_score" and SHAP's
# "engagement"-category feature are two different measurements of
# essentially the same underlying idea. If both happen to cross their
# thresholds on the same turn, saying so twice ("your engagement has
# declined... and your engagement has dropped...") reads as repetitive
# rather than additive. This keeps only the first-mentioned clause per
# rough topic — still zero fabrication, purely a dedup pass over
# clauses already produced above.

def _topic_of(clause: str) -> str:
    lowered = clause.lower()
    if "engagement" in lowered:
        return "engagement"
    if "difficulty following" in lowered:
        return "confusion"
    if "inactiv" in lowered:
        return "inactivity"
    if "study routine" in lowered or "consistent" in lowered:
        return "consistency"
    return lowered


def _dedupe_by_topic(primary: List[str], secondary: List[str]) -> List[str]:
    seen_topics = {_topic_of(c) for c in primary}
    deduped: List[str] = []
    for clause in secondary:
        topic = _topic_of(clause)
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        deduped.append(clause)
    return deduped


# =========================================================
# SITUATION COMPOSITION — ONE sentence, not fragments
# =========================================================
# PHASE 2.4: `human_reason` (still owned by mentor_service.py's
# `_HUMAN_REASON_BY_FOCUS_ID`, unchanged responsibility split) is now
# treated as a noun phrase that completes "Your recent activity
# suggests ___", not as a standalone clause to be glued to others with
# commas. Cognitive/SHAP clauses (already full clauses — see above)
# are then joined onto it with "and", producing one coherent sentence
# a mentor would actually say, e.g.:
#   "Your recent activity suggests increasing signs of fatigue and
#    strain, and your engagement has gradually declined over time."
# instead of the old:
#   "a severe burnout risk pattern, alongside gradually declining
#    engagement"

def _compose_situation(
    human_reason: str,
    extra_clauses: List[str],
    voice: str,
) -> str:
    subject = "Your recent activity" if voice == "student" else "This learner's recent activity"
    sentence = f"{subject} suggests {human_reason}"

    if extra_clauses:
        sentence = f"{sentence}, and " + ", and ".join(extra_clauses)

    return sentence + "."


# =========================================================
# OBSERVATION (replaces question-echoing openers)
# =========================================================
# PHASE 2.4: no longer echoes the student's own words ("You
# mentioned..."). Instead, a single fixed, hand-written observation is
# chosen by `detected_intent` — the same deterministic, categorical
# discipline as the rest of this module. This is intentionally NOT a
# keyword-based paraphrase of the actual question text: with no LLM in
# this path, any attempt to summarize free text keyword-by-keyword
# would risk misreading it. Inferring from the already-classified
# intent category is the safe, deterministic middle ground between
# "echo the words" and "guess at their meaning".

_OBSERVATION_BY_INTENT = {
    "academic": [
        "It sounds like recent academic setbacks have been affecting your confidence.",
        "It sounds like recent coursework has been tougher to stay on top of than usual.",
    ],
    "emotional": [
        "It sounds like things have been feeling heavier than usual lately.",
        "It sounds like this has been a genuinely hard stretch emotionally.",
    ],
    "motivation": [
        "It sounds like it's been harder than usual to stay motivated recently.",
        "It sounds like your motivation has been harder to hold onto lately.",
    ],
    "career": [
        "It sounds like you're weighing some important decisions about what comes next.",
        "It sounds like there's a lot to think through about your next steps.",
    ],
    "general": [
        "It sounds like there's been a lot on your mind recently.",
        "It sounds like there's a fair amount going on for you right now.",
    ],
}


def _opening_observation(
    detected_intent: str,
    question: str,
    persona: Optional[str] = None,
    focus_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    history_depth: int = 0,
) -> Optional[str]:
    if not question or not question.strip():
        return None
    pool = _OBSERVATION_BY_INTENT.get(detected_intent, _OBSERVATION_BY_INTENT["general"])
    return _select_variant(
        pool, "observation", persona, focus_id, risk_level, detected_intent, history_depth
    )


# =========================================================
# RISK -> PLAIN LANGUAGE (never probabilities)
# =========================================================
# PHASE 2.4: new. Translates the categorical risk_level already
# computed elsewhere (confidence.py / context_builder.py) into a fixed
# sentence, so the vague "overall indicators are elevated" phrasing is
# replaced with something a person would actually say. Skipped
# entirely when the focus itself already implies the risk level (e.g.
# "burnout" already means High by construction in
# determine_priority_focus() — repeating it here would be redundant,
# not additive, exactly as the Phase 2.3 code already guarded against).

_RISK_SENTENCE_STUDENT = {
    "High": [
        "You've shown some consistent warning signs across your recent activity.",
        "A few consistent warning signs have shown up across your recent activity.",
    ],
    "Medium": [
        "There are some early signs worth paying attention to before they grow.",
        "A few early signs are worth keeping an eye on before they grow.",
    ],
    "Low": [
        "You're generally doing well, with only minor things to keep an eye on.",
        "Overall you're doing well, with just a couple of small things to watch.",
    ],
}

_RISK_SENTENCE_EDUCATOR = {
    "High": [
        "The learner has shown consistent warning signs across recent learning activities.",
        "Consistent warning signs have shown up across this learner's recent activity.",
    ],
    "Medium": [
        "There are some early warning signs that deserve attention before they grow.",
        "A few early warning signs deserve attention before they grow.",
    ],
    "Low": [
        "The learner is generally progressing well with only minor concerns.",
        "This learner is progressing well overall, with only minor concerns.",
    ],
}

_RISK_REDUNDANT_WITH_FOCUS = {
    ("High", "burnout"),
}


def _risk_sentences(
    risk_level: Optional[str],
    focus_id: str,
    persona: Optional[str] = None,
    detected_intent: Optional[str] = None,
    history_depth: int = 0,
) -> Dict[str, Optional[str]]:
    if (risk_level, focus_id) in _RISK_REDUNDANT_WITH_FOCUS:
        return {"risk_sentence_student": None, "risk_sentence_educator": None}

    signal = (persona, focus_id, detected_intent, history_depth)
    return {
        "risk_sentence_student": _select_variant(
            _RISK_SENTENCE_STUDENT.get(risk_level), "risk_student", *signal
        ),
        "risk_sentence_educator": _select_variant(
            _RISK_SENTENCE_EDUCATOR.get(risk_level), "risk_educator", *signal
        ),
    }


# =========================================================
# MEMORY -> NATURAL ACKNOWLEDGMENT (never fabricated)
# =========================================================
# `memory` is the dict already returned by
# reasoning_layer.derive_persona_memory() (Phase 2.1/2.2 — unchanged
# here). PHASE 2.4: instead of one generic sentence reused for every
# learner with ANY history ("building on what we've already been
# working through"), this now looks at the actual most-recently-used
# action tag and turns IT into plain words (underscores/hyphens ->
# spaces). This references real, already-stored data — nothing new is
# invented — while still reading like it's about THIS learner's
# specific prior support rather than a boilerplate line. If there is
# no real history, both clauses are still None and the renderers omit
# them entirely, exactly as before.

def _humanize_action_tag(tag: str) -> str:
    return tag.replace("_", " ").replace("-", " ").strip()


def _memory_clauses(
    memory: Optional[Dict[str, Any]],
    short_term_context_relevant: bool = True,
) -> Dict[str, Optional[str]]:
    if not short_term_context_relevant:
        return {"memory_clause_student": None, "memory_clause_educator": None}

    if not memory or not memory.get("has_history"):
        return {"memory_clause_student": None, "memory_clause_educator": None}

    used_action_tags = memory.get("used_action_tags") or []
    if not used_action_tags:
        return {"memory_clause_student": None, "memory_clause_educator": None}

    last_tag = _humanize_action_tag(str(used_action_tags[-1]))
    if not last_tag:
        return {"memory_clause_student": None, "memory_clause_educator": None}

    return {
        "memory_clause_student": f"since you've already been working on {last_tag},",
        "memory_clause_educator": f"this continues previous support that focused on {last_tag};",
    }


# =========================================================
# SPRINT B — ACADEMIC RECOMMENDATION CONTEXTUALIZATION
# =========================================================
# Final Backend Sprint B ("Recommendation Relevance"). Problem: the
# recommendation engine already selects the right action and the
# renderer already produces natural wording, but for academic
# questions the rendered sentence never references the specific
# concept just discussed, so a perfectly-grounded action ("shorten
# your next study session") reads as generic and disconnected from
# "difference between BFS and DFS."
#
# This section adds ONE new, purely additive lead-in clause, built the
# exact same way `_memory_clauses()` above already builds its lead-in
# clauses (lowercase start, trailing comma, meant to be folded onto the
# front of an existing sentence via mentor_service.py's
# `_combine_lead()` — unchanged, reused as-is).
#
# HARD CONSTRAINTS:
# - Only ever produces a clause when `detected_intent == "academic"`
#   AND a real `academic_topic` string was passed in. That value is
#   the SAME deterministic `intent_classifier.extract_academic_topic()`
#   output mentor_service.py already computes for every question (it
#   was previously only *used* on the celebration path) — no new
#   extraction, no new data source, nothing inferred from the question
#   text here.
# - If `academic_topic` is falsy, both clauses resolve to `None` and
#   the caller's sentence is returned completely unchanged — same
#   "resolve to None -> automatically skipped" convention every other
#   optional slot in this module already follows.
# - Never touches `determine_priority_focus()`,
#   `recommendation_engine.select_recommendation()`, `human_reason`,
#   or which action was chosen — only the wording of the lead-in
#   clause placed in front of the already-chosen action.
# - Deterministic variant selection reuses the same `signal` tuple
#   (persona, focus_id, risk_level, detected_intent, history_depth)
#   every other wording-variant slot in this file already uses via
#   `_select_variant` — no randomness, no new signal source.
_ACADEMIC_CONTEXT_TEMPLATES = {
    "student": [
        "after reviewing {topic},",
        "now that we've gone over {topic},",
    ],
    "educator": [
        "during the next discussion on {topic},",
        "the next time {topic} comes up,",
    ],
}


def _academic_context_clauses(
    detected_intent: str,
    academic_topic: Optional[str],
    signal: tuple,
) -> Dict[str, Optional[str]]:
    """
    Returns `{"academic_context_student": ..., "academic_context_educator": ...}`.
    Both resolve to `None` unless this is an academic question with a
    real, already-extracted topic — see the module note above for the
    full rationale and hard constraints.
    """
    if detected_intent != "academic" or not academic_topic:
        return {"academic_context_student": None, "academic_context_educator": None}

    topic = academic_topic.strip()
    if not topic:
        return {"academic_context_student": None, "academic_context_educator": None}

    student_template = _select_variant(
        _ACADEMIC_CONTEXT_TEMPLATES["student"], "academic_context_student", *signal
    )
    educator_template = _select_variant(
        _ACADEMIC_CONTEXT_TEMPLATES["educator"], "academic_context_educator", *signal
    )

    return {
        "academic_context_student": (
            student_template.format(topic=topic) if student_template else None
        ),
        "academic_context_educator": (
            educator_template.format(topic=topic) if educator_template else None
        ),
    }


# =========================================================
# PHASE 2.5 — PERSONA VOICE (HOW Aura speaks, not WHAT)
# =========================================================
# Everything above this point decides WHAT Aura talks about: the
# situation sentence is built from human_reason (owned by
# mentor_service.py's `_HUMAN_REASON_BY_FOCUS_ID`, itself driven by
# `determine_priority_focus()`), cognitive/SHAP signals, and the risk
# level. None of that changes in this phase — Priority Focus still
# decides the topic.
#
# What was missing is a persona-level VOICE layer: a fixed opening
# sentence, closing sentence, and encouragement phrasing template per
# persona, plus a persona-specific cap on how many extra cognitive/SHAP
# clauses get folded into the situation sentence (sentence length).
# These four things are looked up ONCE per persona and applied
# regardless of which focus_id or risk_level this particular turn
# produced — i.e. Priority Focus can change WHAT the situation sentence
# talks about, but it can never change WHICH persona voice wraps around
# it. This is deliberately a plain lookup table (no computation, no
# randomness), the same "hand-written phrase per known category"
# discipline as `_PERSONA_ACTIONS` / `_PSYCHOLOGICAL_FRAMEWORKS` in
# prompt_builder.py and `_FEW_SHOT_BANK` in reasoning_layer.py — this
# module intentionally does not import those (to avoid a circular
# import, same reasoning reasoning_layer.py's own docstring gives for
# not importing prompt_builder.py) and instead keeps its own small,
# self-contained voice table for the one thing it renders: the final
# recommendation strings.
#
# Applied to the STUDENT-facing recommendation only. The educator
# recommendation stays the professional case-note register Phase 2.4
# established (item #8 of that phase's brief) — a case note documents
# the learner for a staff audience, it is not "Aura's voice" in the
# same sense the direct-to-student message is, so it is not re-toned
# per persona here. Persona already differentiates the educator note's
# CONTENT (via `_PERSONA_ACTIONS`'s educator action and the situation
# sentence itself) — Phase 2.5 only adds a further, distinct HOW layer
# on top of the student-facing message.

# SPRINT 3: each of "opening" / "encouragement_template" / "closing"
# is now a small (2-variant) pool instead of one fixed string, so the
# same persona doesn't produce byte-identical voice sentences on every
# turn. `max_extra_clauses` is unchanged (a plain int, not wording).
# Variant [0] in every pool below is the exact original Phase 2.5
# string.
_PERSONA_VOICE = {
    "Burnout Pattern": {
        "opening": [
            "Let's slow down for a moment.",
            "Let's take this one slow step at a time.",
        ],
        "encouragement_template": [
            "There's no rush here — {action} is one gentle step you can "
            "take right now.",
            "Whenever it feels manageable, {action} is a gentle place to "
            "start.",
        ],
        "closing": [
            "Rest counts as real progress too.",
            "Taking it slow is still moving forward.",
        ],
        "max_extra_clauses": 1,
    },
    "Anxiety-Spike Learner": {
        "opening": [
            "What you're feeling right now makes complete sense.",
            "It makes sense that this would feel like a lot right now.",
        ],
        "encouragement_template": [
            "When you're ready, {action} — a small, steady step is enough.",
            "Whenever you feel ready, {action}; one steady step is enough.",
        ],
        "closing": [
            "You're handling more than you're giving yourself credit for.",
            "You're doing better with this than it probably feels like.",
        ],
        "max_extra_clauses": 1,
    },
    "Passive Watcher": {
        "opening": [
            "Let's turn this into something you actively do, not just watch.",
            "Let's turn this from watching into doing.",
        ],
        "encouragement_template": [
            "Here's a simple way to jump in right now — {action}.",
            "Here's a straightforward way to get moving — {action}.",
        ],
        "closing": [
            "Small actions build real momentum — let's keep it going.",
            "Momentum starts with one small action — let's build it.",
        ],
        "max_extra_clauses": 2,
    },
    "Silent Isolator": {
        "opening": [
            "I'm really glad you reached out.",
            "I'm glad you decided to reach out about this.",
        ],
        "encouragement_template": [
            "Here's one small, low-pressure step forward — {action}.",
            "Here's a small, low-pressure next step — {action}.",
        ],
        "closing": [
            "You don't have to figure this out alone.",
            "You're not expected to work through this by yourself.",
        ],
        "max_extra_clauses": 1,
    },
    "Last-Minute Survivor": {
        "opening": [
            "Let's map this out clearly so nothing catches you off guard.",
            "Let's get a clear plan down so nothing sneaks up on you.",
        ],
        "encouragement_template": [
            "Here's the plan — {action}.",
            "Here's the next concrete step — {action}.",
        ],
        "closing": [
            "Stick to the plan, one step at a time, and you'll be ahead "
            "of the deadline.",
            "Follow the plan one step at a time and you'll stay ahead of "
            "the deadline.",
        ],
        "max_extra_clauses": 2,
    },
    "Consistent Learner": {
        "opening": [
            "You've clearly been putting in solid, consistent effort.",
            "You've been putting in consistently solid effort.",
        ],
        "encouragement_template": [
            "As your next challenge, {action}.",
            "For your next stretch goal, {action}.",
        ],
        "closing": [
            "Keep pushing — you're more than ready for the next level.",
            "You're more than ready for the next level — keep pushing.",
        ],
        "max_extra_clauses": 2,
    },
}

_DEFAULT_PERSONA_VOICE = {
    "opening": [
        "Here's what I'm noticing.",
        "Here's what stands out to me right now.",
    ],
    "encouragement_template": [
        "Given all this, choosing to {action} could really help right now.",
        "With all that in mind, {action} could really help right now.",
    ],
    "closing": [
        "Every small step forward counts.",
        "Every bit of forward progress counts.",
    ],
    "max_extra_clauses": 2,
}


def _persona_voice(persona: Optional[str]) -> Dict[str, Any]:
    return _PERSONA_VOICE.get(persona, _DEFAULT_PERSONA_VOICE)


# =========================================================
# PHASE 2.6 — PERSONA CONVERSATION STRATEGY (ORDER + EMPHASIS)
# =========================================================
# Phases 2.3-2.5 decide WHAT is said (situation, risk framing, memory,
# action) and, per persona, HOW individual sentences are worded
# (opening/closing/encouragement phrasing). What was still identical
# across every persona was the CONVERSATION SHAPE: every recommendation
# walked Opening -> Observation -> Interpretation -> Risk ->
# Recommendation -> Closing, in that fixed order, every time. A
# burnt-out learner and a thriving one were being mentored with the
# same six-step script, just with different words dropped into the
# same slots.
#
# This adds a persona -> ORDERED LIST OF SLOTS lookup. A "slot" names
# a sentence that already exists elsewhere in this module or in the
# renderer (mentor_service.py's `_render_student_recommendation`):
#   "opening"     -> persona_opening
#   "observation" -> opening_observation
#   "situation"   -> situation_student
#   "risk"        -> risk_sentence_student
#   "action"      -> persona_encouragement_template.format(action=...)
#   "closing"     -> persona_closing
#
# No slot introduces new evidence, a new sentence, or a new fact about
# the learner -- every slot's CONTENT is still produced by the exact
# same logic as before (Priority Focus for topic, Phase 2.4 for
# wording, Phase 2.5 for persona voice). This layer only decides:
#   1. WHICH of those six sentences a persona's mentoring style
#      actually uses (a slot simply absent from the list is omitted --
#      e.g. Burnout Pattern never states the risk sentence outright,
#      because piling on a warning contradicts "reduce pressure"), and
#   2. In WHAT ORDER they're said (e.g. Passive Watcher states the
#      risk BEFORE the situation -- leading with the consequence to
#      challenge passive behaviour -- while every other persona leads
#      with the softer situation sentence first).
#
# Mapping to the Phase 2.6 brief's six mentoring strategies:
#   Burnout Pattern       : Validate feelings (situation) -> reduce
#                           pressure (no risk sentence) -> one small
#                           step (action) -> gentle reassurance (closing).
#   Anxiety-Spike Learner : Normalize (observation) -> ground the
#                           learner (situation) -> reduce fear (no risk
#                           sentence) -> recommendation -> calming close.
#   Passive Watcher       : Challenge (risk stated FIRST) -> motivate
#                           (situation) -> recommendation -> momentum
#                           closing.
#   Silent Isolator       : Connection (opening+observation) ->
#                           normalize asking for help (situation, no
#                           risk sentence) -> recommendation -> warm
#                           closing.
#   Last-Minute Survivor  : Acknowledge procrastination (situation) ->
#                           planning mindset (risk framed as the reason
#                           to plan) -> recommendation (the plan) ->
#                           accountability closing.
#   Consistent Learner    : Celebrate progress (situation only, no risk
#                           sentence to undercut the praise) -> stretch
#                           challenge (recommendation) -> positive close.
#
# `recommendation_engine.py`'s selection and `determine_priority_focus`
# are never consulted or altered here -- this table only reorders/
# includes-or-omits sentences that were already going to be produced.
# Deterministic (plain dict lookup, no randomness) and additive: any
# persona not listed (including "Unknown Persona") falls back to
# `_DEFAULT_CONVERSATION_STRATEGY`, which reproduces the exact Opening
# -> Observation -> Situation -> Risk -> Action -> Closing order the
# renderer already used before this phase -- so behaviour for any
# unmapped persona is unchanged.

_DEFAULT_CONVERSATION_STRATEGY = [
    "opening", "observation", "situation", "risk", "action", "closing",
]

_PERSONA_CONVERSATION_STRATEGY = {
    # Very high empathy, very low pressure -> lead the interpretation
    # with an explicit empathy beat, soften the hand-off into the
    # action with a low-pressure transition, and reassure right after
    # the action that the (unchanged) recommended step is enough.
    "Burnout Pattern": [
        "opening", "empathy", "situation", "transition", "action",
        "reassurance", "closing",
    ],

    # High reassurance, normalize emotions, reduce catastrophic
    # thinking -> empathy normalizes the feeling right after the
    # observation, reassurance comes BEFORE the transition/action so
    # the catastrophic-thinking reduction happens before any ask.
    "Anxiety-Spike Learner": [
        "opening", "observation", "empathy", "situation", "reassurance",
        "transition", "action", "closing",
    ],

    # Lower empathy, higher activation, challenge gently -> the
    # "empathy"/"reassurance" slots are intentionally NOT used (same
    # absent-slot mechanism as Phase 2.6 omitting "risk" for Burnout);
    # risk still leads to challenge, then a brisk transition straight
    # into the action, matching "encourage immediate action".
    "Passive Watcher": ["opening", "risk", "situation", "transition", "action", "closing"],

    # Warm, relationship-focused -> empathy right after the
    # observation, then a connection-oriented transition into the
    # action, then reassurance that asking for help is not a burden.
    "Silent Isolator": [
        "opening", "observation", "empathy", "situation", "transition",
        "action", "reassurance", "closing",
    ],

    # Structured, practical, slightly more directive -> no empathy or
    # reassurance beat (a case-note-adjacent, plan-first register);
    # "transition" here reads as "let's lock in the plan", right before
    # the action, matching "planning-focused".
    "Last-Minute Survivor": [
        "opening", "situation", "risk", "transition", "action", "closing",
    ],

    # Celebratory, growth mindset, stretch goals -> no separate empathy
    # beat (the celebratory situation sentence already carries warmth);
    # transition frames the action as an earned stretch goal, and
    # reassurance afterward reinforces that the learner has already
    # proven they can handle it (positive reinforcement).
    "Consistent Learner": [
        "opening", "situation", "transition", "action", "reassurance",
        "closing",
    ],
}


def get_conversation_strategy(persona: Optional[str]) -> List[str]:
    """
    PHASE 2.6 ADDITION. Returns the ordered list of slot names this
    persona's mentoring strategy uses (see module note above). Falls
    back to `_DEFAULT_CONVERSATION_STRATEGY` -- the original fixed
    Opening/Observation/Situation/Risk/Action/Closing order -- for any
    persona not explicitly mapped, so unmapped/unknown personas render
    exactly as they did before this phase (full backward compatibility).

    PHASE 2.7 NOTE: the lists below now also reference three new slot
    names -- "empathy", "transition", "reassurance" -- whose sentence
    content comes from `get_communication_profile()` below, not from
    this function. This function still only returns slot NAMES; it has
    no idea what text a slot resolves to, so it needed no change beyond
    this table's contents.
    """
    return list(_PERSONA_CONVERSATION_STRATEGY.get(persona, _DEFAULT_CONVERSATION_STRATEGY))


# =========================================================
# PHASE 3.3 — ADAPTIVE CONVERSATION PLANNING (ORDERING ONLY)
# =========================================================
# `apply_conversation_goal_ordering()` is purely a REORDERING of the
# slot names `get_conversation_strategy()` already returned for this
# persona. It never adds, removes, renames, or invents a slot, never
# touches `recommendation_engine.py` / `select_recommendation()`'s
# choice of WHICH action was recommended, and never changes
# `determine_priority_focus()`'s selection. It answers only: "given
# this turn's conversational goal, should any of THIS persona's
# already-chosen slots be read in a different order?"
#
# If `conversation_goal` is None, empty, or not one of the goals
# recognized below, the input list is returned unchanged (same order,
# same contents) -- so any caller that doesn't pass a goal, or that
# hasn't adopted Phase 3.3 at all, is completely unaffected.
#
# For the two goals that mean "the learner's emotional state must be
# addressed before any action/transition" ("Reduce pressure" and
# "Stabilize emotion first"), any "empathy" / "reassurance" slots the
# persona's OWN strategy already includes are moved to sit right after
# the opening/observation slots and before everything else -- so a
# persona whose strategy already reassures early is unaffected, and a
# persona whose strategy reassures late only gets it pulled forward
# when THIS turn's conversational goal calls for it. No other goal
# changes ordering at all in this phase.
_STABILIZING_CONVERSATION_GOALS = {"Reduce pressure", "Stabilize emotion first"}


def apply_conversation_goal_ordering(
    strategy: List[str],
    conversation_goal: Optional[str] = None,
) -> List[str]:
    """
    Reorders (never modifies the contents of) an existing
    `get_conversation_strategy(persona)` result based on this turn's
    `conversation_goal` (see prompt_builder.determine_conversation_goal).
    Deterministic, side-effect-free, backward compatible: omitting
    `conversation_goal` (the default, `None`) returns `strategy`
    unchanged.
    """
    if not conversation_goal or conversation_goal not in _STABILIZING_CONVERSATION_GOALS:
        return list(strategy)

    lead_slots = [slot for slot in strategy if slot in ("opening", "observation")]
    pulled_forward = [slot for slot in strategy if slot in ("empathy", "reassurance")]
    remaining = [
        slot for slot in strategy
        if slot not in lead_slots and slot not in pulled_forward
    ]

    return lead_slots + pulled_forward + remaining


# =========================================================
# PHASE 2.7 — PERSONA COMMUNICATION PROFILE (EMOTIONAL DEPTH)
# =========================================================
# Phase 2.6 gave each persona a different SHAPE (which sentences, in
# what order). Every one of those sentences, though, was still drawn
# from the same emotional register: fond but fairly even-keeled. A
# "Very high empathy / very low pressure" mentor and a "lower empathy /
# higher activation" mentor were still saying structurally-similar
# things, just in a different sequence.
#
# This section adds a persona -> COMMUNICATION PROFILE lookup. Each
# profile has two parts:
#
#   1. Six descriptive DIALS (documentation/introspection only, not
#      directly rendered): empathy_level, directness, reassurance_level,
#      motivational_energy, pacing, coaching_style. These name, in one
#      place, the intended emotional posture the phrases below were
#      hand-written to express -- useful for review/auditing ("does
#      Burnout Pattern's phrase bank actually read as very-high-empathy,
#      very-low-pressure?") without being three more strings the
#      renderer has to interpret.
#   2. Three NEW, UNIQUE, hand-written phrases per persona --
#      `empathy_phrase`, `transition_phrase`, `reassurance_phrase` --
#      that are honestly new sentences (not reused/reworded from
#      `_PERSONA_VOICE`'s opening/closing/encouragement_template), so a
#      persona that uses all three slots doesn't repeat itself. A
#      persona whose profile calls for LOWER empathy/reassurance (e.g.
#      Passive Watcher's "lower empathy, higher activation") still gets
#      real phrases here for completeness/consistency of the data
#      model, but its entry in `_PERSONA_CONVERSATION_STRATEGY` simply
#      does not include the "empathy"/"reassurance" slots -- exactly
#      the same "slot present vs. absent" mechanism Phase 2.6 already
#      established for "risk". Nothing here overrides that mechanism;
#      it reuses it.
#
# None of this touches `human_reason`, `situation_student`, the risk
# sentence, the action string, or ANY evidence-bearing content. These
# are pure tone/register phrases -- the same "fixed, hand-written
# phrase per known category, never fabricated" discipline the rest of
# this module follows -- inserted around the untouched recommendation,
# never replacing or paraphrasing it.

_DEFAULT_COMMUNICATION_PROFILE = {
    "empathy_level": "medium",
    "directness": "medium",
    "reassurance_level": "medium",
    "motivational_energy": "medium",
    "pacing": "medium",
    "coaching_style": "balanced",
    "empathy_phrase": None,
    "transition_phrase": None,
    "reassurance_phrase": None,
}

_PERSONA_COMMUNICATION_PROFILE = {
    "Burnout Pattern": {
        "empathy_level": "very_high",
        "directness": "very_low",
        "reassurance_level": "high",
        "motivational_energy": "low",
        "pacing": "slow",
        "coaching_style": "permission_based",
        "empathy_phrase": [
            "It's okay if progress feels slow today.",
            "It's okay if today just feels slow.",
        ],
        "transition_phrase": [
            "If it feels manageable, here's a small idea:",
            "If you're up for it, here's one small idea:",
        ],
        "reassurance_phrase": [
            "You don't have to earn rest — it's already enough.",
            "Rest doesn't have to be earned — it's already enough.",
        ],
    },
    "Anxiety-Spike Learner": {
        "empathy_level": "high",
        "directness": "low",
        "reassurance_level": "very_high",
        "motivational_energy": "low",
        "pacing": "slow",
        "coaching_style": "calming_normalizing",
        "empathy_phrase": [
            "Your mind might be jumping to the worst case, and that's a "
            "very human response.",
            "It's natural for your mind to jump to the worst case right "
            "now.",
        ],
        "transition_phrase": [
            "Here's one grounded step, at whatever pace works:",
            "Here's one grounded step, whenever you're ready for it:",
        ],
        "reassurance_phrase": [
            "This feeling will ease, and you don't have to solve "
            "everything today.",
            "This will ease with time, and today doesn't have to solve "
            "everything.",
        ],
    },
    "Passive Watcher": {
        "empathy_level": "low",
        "directness": "high",
        "reassurance_level": "low",
        "motivational_energy": "high",
        "pacing": "fast",
        "coaching_style": "challenge_and_activate",
        "empathy_phrase": [
            "I know watching from the sidelines can feel safer.",
            "Watching from the sidelines can feel like the safer option.",
        ],
        "transition_phrase": [
            "Time to swap watching for doing:",
            "Let's swap watching for doing:",
        ],
        "reassurance_phrase": [
            "One step is all it takes to break the pattern.",
            "Breaking the pattern only takes one step.",
        ],
    },
    "Silent Isolator": {
        "empathy_level": "high",
        "directness": "low",
        "reassurance_level": "high",
        "motivational_energy": "medium",
        "pacing": "medium",
        "coaching_style": "relationship_focused",
        "empathy_phrase": [
            "Reaching out like this takes real courage.",
            "It takes real courage to reach out like this.",
        ],
        "transition_phrase": [
            "Here's one gentle way to stay connected:",
            "Here's a gentle way to stay connected:",
        ],
        "reassurance_phrase": [
            "Asking for support is a strength, not a burden.",
            "Asking for support is a strength — never a burden.",
        ],
    },
    "Last-Minute Survivor": {
        "empathy_level": "medium",
        "directness": "high",
        "reassurance_level": "medium",
        "motivational_energy": "medium",
        "pacing": "fast",
        "coaching_style": "structured_directive",
        "empathy_phrase": [
            "Leaving things late doesn't mean you don't care about the outcome.",
            "Running late on this doesn't mean the outcome doesn't matter to you.",
        ],
        "transition_phrase": [
            "Let's lock in the next concrete step:",
            "Let's pin down the next concrete step:",
        ],
        "reassurance_phrase": [
            "A clear plan now puts you back in control.",
            "Having a clear plan puts you back in the driver's seat.",
        ],
    },
    "Consistent Learner": {
        "empathy_level": "medium",
        "directness": "medium",
        "reassurance_level": "medium",
        "motivational_energy": "very_high",
        "pacing": "medium",
        "coaching_style": "celebratory_growth",
        "empathy_phrase": [
            "Consistency like yours doesn't happen by accident.",
            "Showing up this consistently doesn't happen by accident.",
        ],
        "transition_phrase": [
            "Since you're ready for more, here's a stretch goal:",
            "You're ready for more, so here's a stretch goal:",
        ],
        "reassurance_phrase": [
            "You've already proven you can handle this.",
            "You've already shown you can handle this.",
        ],
    },
}


def get_communication_profile(persona: Optional[str]) -> Dict[str, Any]:
    """
    PHASE 2.7 ADDITION. Returns this persona's communication profile —
    the six descriptive dials plus the three unique phrases described
    in the module note above. Falls back to
    `_DEFAULT_COMMUNICATION_PROFILE` (all phrases `None`) for any
    persona not explicitly mapped, so an unmapped/unknown persona's
    `_render_student_recommendation` output is unaffected by this
    phase — the new "empathy"/"transition"/"reassurance" slots simply
    resolve to nothing and are skipped, exactly like an absent slot in
    `_PERSONA_CONVERSATION_STRATEGY`.
    """
    return dict(_PERSONA_COMMUNICATION_PROFILE.get(persona, _DEFAULT_COMMUNICATION_PROFILE))


# =========================================================
# PUBLIC ENTRY POINT
# =========================================================

def build_recommendation_rationale(
    human_reason: str,
    detected_intent: str,
    risk_level: Optional[str],
    focus_id: str,
    cognitive_state: Optional[Dict[str, Any]],
    shap_explanation: Optional[Dict[str, Any]],
    question: str,
    memory: Optional[Dict[str, Any]],
    persona: str = "Unknown Persona",
    conversation_goal: Optional[str] = None,
    short_term_context_relevant: bool = True,
    academic_topic: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Builds the natural-language rationale fragments described in the
    module docstring. `human_reason` is the SAME string
    mentor_service.py's `_HUMAN_REASON_BY_FOCUS_ID` already produces
    (Sprint 6, reworded in Phase 2.4) — this function extends it with
    cognitive/SHAP/memory/risk observations rather than recomputing it,
    so the underlying evidence-to-prose mapping stays owned in one
    place.

    PHASE 2.5 ADDITION: `persona` (new, defaulted to "Unknown Persona"
    so any existing caller that doesn't pass it keeps working
    unmodified) selects a fixed voice profile — opening sentence,
    closing sentence, encouragement phrasing, and how many extra
    cognitive/SHAP clauses the situation sentence is allowed to carry.
    This voice is looked up once and applied regardless of focus_id or
    risk_level, so Priority Focus keeps deciding WHAT the situation
    sentence discusses while persona alone decides HOW it's delivered.

    PHASE 3.3 ADDITION (Adaptive Conversation Planning): `conversation_goal`
    (new, defaulted to `None` so any existing caller that doesn't pass
    it keeps getting the exact same `conversation_strategy` ordering as
    before) is used ONLY to reorder the persona's already-selected
    conversation-strategy slots via `apply_conversation_goal_ordering()`
    — see that function's docstring. It never changes `human_reason`,
    never changes which cognitive/SHAP clauses are eligible, never
    touches `determine_priority_focus()` or `recommendation_engine.py`'s
    selection, and never changes the CONTENTS of `conversation_strategy`
    — only, for two specific goals, the order of slots already present.

    SPRINT 2 ADDITION: `short_term_context_relevant` (new, defaulted to
    `True` so any existing caller that doesn't pass it keeps getting the
    exact same memory-clause behaviour as before) is passed straight
    through to `_memory_clauses()`. When `False`, both memory clauses
    resolve to `None` — the short-term chat-history callback is
    suppressed for this turn — while every other field returned below
    (situation, risk, persona voice, conversation strategy, communication
    profile) is computed exactly as it already was. This is the only
    effect of the new parameter.

    SPRINT B ADDITION (Recommendation Relevance): `academic_topic` (new,
    defaulted to `None` so any existing caller that doesn't pass it
    keeps working unmodified) is passed straight through to
    `_academic_context_clauses()`. It only ever produces a non-`None`
    clause when `detected_intent == "academic"` and a real topic string
    was given — see that function's module note. It never changes
    `human_reason`, `situation_student`/`situation_educator`, which
    cognitive/SHAP signals are eligible, or any WHAT-level decision
    already made above; it is a new, independent, additive pair of keys
    only.

    Internal-only: the returned dict is consumed by mentor_service.py's
    renderers to build the final recommendation strings. It is never
    itself returned to an API consumer.
    """
    memory = memory or {}
    voice = _persona_voice(persona)
    communication = get_communication_profile(persona)

    # SPRINT 3: how many actions this learner has already been given
    # (unchanged data source -- `derive_persona_memory()`'s
    # `used_action_tags`, already present on `memory`). Used only as
    # one more deterministic signal for wording-variant selection
    # below; it does not affect memory_clauses, risk gating, or any
    # other WHAT-level decision.
    history_depth = len(memory.get("used_action_tags") or [])
    signal = (persona, focus_id, risk_level, detected_intent, history_depth)

    cognitive_student = _describe_cognitive_state(cognitive_state, voice="student")
    cognitive_educator = _describe_cognitive_state(cognitive_state, voice="educator")
    shap_student = _dedupe_by_topic(cognitive_student, _describe_shap(shap_explanation, voice="student"))
    shap_educator = _dedupe_by_topic(cognitive_educator, _describe_shap(shap_explanation, voice="educator"))

    # PHASE 2.5: persona caps how many extra clauses the STUDENT-facing
    # situation sentence carries (sentence-length control). The
    # educator case note stays uncapped/complete — professional
    # documentation isn't shortened for tone.
    extra_student = (cognitive_student + shap_student)[: voice["max_extra_clauses"]]
    extra_educator = cognitive_educator + shap_educator

    situation_student = _compose_situation(human_reason, extra_student, voice="student")
    situation_educator = _compose_situation(human_reason, extra_educator, voice="educator")

    risk_sentences = _risk_sentences(risk_level, focus_id, persona, detected_intent, history_depth)
    memory_clauses = _memory_clauses(memory, short_term_context_relevant)
    academic_context = _academic_context_clauses(detected_intent, academic_topic, signal)

    return {
        "opening_observation": _opening_observation(
            detected_intent, question, persona, focus_id, risk_level, history_depth
        ),
        "situation_student": situation_student,
        "situation_educator": situation_educator,
        "risk_sentence_student": risk_sentences["risk_sentence_student"],
        "risk_sentence_educator": risk_sentences["risk_sentence_educator"],
        "memory_clause_student": memory_clauses["memory_clause_student"],
        "memory_clause_educator": memory_clauses["memory_clause_educator"],

        # SPRINT B ADDITION (Recommendation Relevance) -- purely
        # additive keys. `None`/`None` for every non-academic intent
        # and for any academic question where no topic was
        # deterministically extracted, so an older renderer or an
        # unmodified call site is completely unaffected. See
        # `_academic_context_clauses()`'s module note above.
        "academic_context_student": academic_context["academic_context_student"],
        "academic_context_educator": academic_context["academic_context_educator"],
        # PHASE 2.5 — persona voice (student-facing only; see the
        # module-level note above this function for why the educator
        # note is intentionally excluded). SPRINT 3: each field is now
        # deterministically selected from a small pool via `signal`
        # instead of being the pool's only entry.
        "persona_opening": _select_variant(voice["opening"], "persona_opening", *signal),
        "persona_encouragement_template": _select_variant(
            voice["encouragement_template"], "persona_encouragement", *signal
        ),
        "persona_closing": _select_variant(voice["closing"], "persona_closing", *signal),

        # PHASE 2.6 ADDITION -- purely additive key. The ordered list
        # of slot names this persona's mentoring strategy uses (see
        # the module note above `get_conversation_strategy`). Existing
        # callers that only read the fields already present above are
        # completely unaffected; only mentor_service.py's Phase 2.6
        # renderer update reads this new key.
        #
        # PHASE 3.3 UPDATE: the list is now passed through
        # `apply_conversation_goal_ordering()`, which is a no-op
        # (returns the exact same order) whenever `conversation_goal`
        # is `None` or not one of the two recognized stabilizing goals
        # -- see that function's docstring for the full rationale.
        "conversation_strategy": apply_conversation_goal_ordering(
            get_conversation_strategy(persona), conversation_goal
        ),

        # PHASE 2.7 ADDITION -- purely additive keys. Three unique,
        # hand-written phrases (empathy / transition / reassurance)
        # from this persona's communication profile (see
        # `get_communication_profile` module note above), plus the
        # profile's descriptive dials for auditing/documentation.
        # `None` for any persona without a defined profile, so an
        # unmapped persona's rendered output is byte-for-byte unchanged
        # from before this phase (the corresponding slots simply
        # resolve to nothing and are skipped by the renderer).
        "communication_empathy_phrase": _select_variant(
            communication["empathy_phrase"], "empathy", *signal
        ),
        "communication_transition_phrase": _select_variant(
            communication["transition_phrase"], "transition", *signal
        ),
        "communication_reassurance_phrase": _select_variant(
            communication["reassurance_phrase"], "reassurance", *signal
        ),
        "communication_profile": {
            "empathy_level": communication["empathy_level"],
            "directness": communication["directness"],
            "reassurance_level": communication["reassurance_level"],
            "motivational_energy": communication["motivational_energy"],
            "pacing": communication["pacing"],
            "coaching_style": communication["coaching_style"],
        },
    }
