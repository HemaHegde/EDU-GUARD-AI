"""
recommendation_engine.py

PHASE 2.2 ADDITION (Persona-aware Recommendation Selection Engine).

SCOPE / HARD CONSTRAINTS FOR THIS PHASE
-----------------------------------------
- No frontend changes.
- No API redesign, no database changes, no reasoning-pipeline redesign.
- `determine_priority_focus()` (prompt_builder.py) is NOT touched —
  this module only consumes its output (`focus["id"]`), the same way
  mentor_service.py's fallback-recommendation path already did.
- The Student Renderer (`_render_student_recommendation`) and Educator
  Renderer (`_render_educator_recommendation`) in mentor_service.py are
  NOT modified. This module only decides WHICH action those renderers
  are handed; the renderers still control HOW that action is worded.

PROBLEM THIS SOLVES
--------------------
Before this phase, every persona received exactly one fixed
student_action / educator_action per priority-focus id, straight from
`determine_priority_focus()` / `_PERSONA_ACTIONS`
(prompt_builder.py). The wording differed by renderer, but the
underlying intervention never varied turn to turn, and persona only
changed phrasing, not substance.

WHAT THIS MODULE ADDS
-----------------------
1. Recommendation Libraries — multiple intervention CANDIDATES per
   (persona, priority-focus) pair, instead of one fixed action.
   Organized as plain data (dicts of dicts of lists), not hardcoded
   into mentor_service.py.
2. select_recommendation() — one deterministic helper that narrows
   those candidates down to exactly one, using:
     - priority_focus  (narrows to the relevant intervention family —
       e.g. only burnout-focused candidates when focus == "burnout")
     - detected_intent (narrows further within that family — academic /
       motivation / emotional / career framing)
     - risk_level      (biases toward intervention-oriented candidates
       at High risk, growth-oriented at Low risk)
     - used_action_tags (avoids repeating an already-used intervention;
       falls back gracefully to the highest-priority one if every
       candidate has already been used)
   No randomness anywhere — same inputs always produce the same
   output.

WHICH (persona, focus_id) PAIRS ARE ACTUALLY REACHABLE
---------------------------------------------------------
`determine_priority_focus()`'s own branching (prompt_builder.py,
unchanged) means not every combination can occur:
  - focus == "enrichment" only ever fires for persona == "Consistent
    Learner" (guardrail #1).
  - focus == "burnout" only ever fires for persona == "Burnout
    Pattern" (guardrail #3).
  - focus in ("confusion", "engagement", "inactivity",
    "persona_default") can fire for ANY persona (score/SHAP-based, not
    persona-gated).
This module's library mirrors that reality: "burnout" and
"enrichment" only have persona-specific entries for the one persona
that can reach them, while "confusion" / "engagement" / "inactivity" /
"persona_default" have persona-specific entries for all six known
personas. A generic, persona-agnostic library backstops every focus id
regardless, so lookups never come back empty even for an unrecognized
or "Unknown Persona" value.

DATA SHAPE
-----------
Each candidate is a small dict:
    {
        "tag": str,        # stable id, matches reasoning_layer.py's
                            # _KNOWN_ACTION_TAGS vocabulary where the
                            # intervention already has a recognized tag,
                            # so used_action_tags (derived from real
                            # past replies) can actually suppress it.
        "intent": str,      # one of "academic" | "motivation" |
                            # "emotional" | "career" | "general"
        "risk": str,         # one of "high" | "medium" | "low" | "any"
        "student": str,      # student-facing action fragment
        "educator": str,     # educator-facing action fragment
    }

`tag` values are intentionally reused across candidates that describe
the same real-world intervention worded for different foci/personas
(e.g. "recovery_break" appears for both Burnout Pattern and the
generic library), because they ARE the same intervention — this keeps
`used_action_tags` suppression meaningful instead of every candidate
having a unique tag that never matches anything from chat history.
"""

from typing import Any, Dict, List, Optional

# =========================================================
# INTENT / RISK NORMALIZATION
# =========================================================

_KNOWN_INTENTS = ("academic", "motivation", "emotional", "career")


def _normalize_intent(detected_intent: Optional[str]) -> str:
    """
    intent_classifier.classify_intent() also returns "general" (its
    backward-compatibility catch-all). Treated here identically to any
    other unrecognized value: "general" means "don't narrow by intent",
    not a fifth intervention family.
    """
    if detected_intent in _KNOWN_INTENTS:
        return detected_intent
    return "general"


def _risk_tier(risk_level: Optional[str]) -> str:
    """
    Maps context.risk_level onto the three tiers item #6 of the brief
    describes. "Medium" is the only value that maps to "medium";
    everything else (Low, Unknown, None, or any unrecognized value)
    maps to "low" so a missing/unknown risk signal never accidentally
    behaves like "High" (more intervention-oriented) — the safer
    default when risk is unclear is the gentler, growth-oriented end.
    """
    if risk_level == "High":
        return "high"
    if risk_level == "Medium":
        return "medium"
    return "low"


def _c(tag: str, intent: str, risk: str, student: str, educator: str) -> Dict[str, str]:
    return {"tag": tag, "intent": intent, "risk": risk, "student": student, "educator": educator}


# =========================================================
# GENERIC LIBRARY (persona-agnostic backstop)
# =========================================================
# Guarantees every focus id has SOME candidates even for a persona with
# no dedicated entry (e.g. "Unknown Persona", or a persona that has not
# been given focus-specific entries below). Never empty, never removed.

_GENERIC_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "confusion": [
        _c("clarify_concept", "academic", "high",
           "ask for the specific step or concept that's unclear to be broken down before moving on",
           "consider re-explaining this topic or providing a worked example — confusion score is elevated"),
        _c("concept_review", "academic", "medium",
           "revisit the last worked example one more time before attempting a new problem",
           "share a supplementary worked example targeting the same sub-skill"),
        _c("small_wins", "motivation", "low",
           "re-attempt just one already-seen practice question to rebuild confidence before new material",
           "note the specific sub-topic causing confusion for a targeted follow-up"),
        _c("stress_reduction", "emotional", "any",
           "take a short breather before re-reading the confusing part — confusion often eases with a clear head",
           "acknowledge the difficulty before re-teaching; frame it as normal, not a setback"),
    ],
    "burnout": [
        _c("reduce_workload", "emotional", "high",
           "reduce your workload this week and schedule one real recovery break before your next study session",
           "flag for a workload/wellness check-in; consider a short extension or reduced load"),
        _c("recovery_break", "emotional", "high",
           "block out one real recovery break today with no coursework attached to it",
           "monitor wellbeing signals closely this week; a recovery gap is warranted"),
        _c("one_task_planning", "academic", "medium",
           "pick exactly one task for today and stop there — no stacking additional work on top",
           "consider deadline flexibility so this student isn't compounding tasks under pressure"),
        _c("habit_building", "motivation", "low",
           "once the immediate load eases, rebuild with a small, sustainable daily routine",
           "check in briefly once the acute period passes to help re-establish a sustainable pace"),
    ],
    "engagement": [
        _c("interactive_task", "academic", "medium",
           "do one small interactive action now (answer a quiz question, post one comment) instead of passive review",
           "prompt with an interactive task rather than more reading or video material — engagement score is low"),
        _c("confidence", "motivation", "low",
           "pick the topic you feel most confident about and engage with just that one first",
           "offer a low-stakes, confidence-building task before anything more demanding"),
        _c("forum_question", "emotional", "any",
           "ask one question in the course forum or to a classmate this week, even a small one",
           "reach out proactively — this student is unlikely to initiate contact themselves"),
        _c("networking", "career", "low",
           "connect one topic you're studying to a project or role you're interested in, then explore that angle",
           "suggest a relevant project or peer group that could re-spark engagement"),
    ],
    "inactivity": [
        _c("resume_activity", "academic", "high",
           "log in and complete one small piece of coursework today to break the inactivity streak, then build back up gradually",
           "reach out directly — inactivity is the strongest measured driver of this student's risk"),
        _c("if_then_plan", "motivation", "medium",
           "set a specific If-Then plan (e.g. 'after dinner, review one topic') to restart a regular rhythm",
           "consider a scaffolded reminder or interim checkpoint to re-establish contact"),
        _c("self_care", "emotional", "any",
           "if something outside coursework has been getting in the way, a short check-in with support staff can help before catching up academically",
           "consider whether a wellbeing check-in, not just an academic nudge, is the right first outreach"),
        _c("projects", "career", "low",
           "pick back up with a project-based task — building something can be an easier re-entry point than review",
           "suggest re-entry through a project or applied task rather than pure catch-up review"),
    ],
    "enrichment": [
        _c("enrichment", "academic", "any",
           "take on an enrichment activity or advanced topic — current performance supports it",
           "consider this student for a peer-mentoring or leadership opportunity"),
        _c("habit_building", "motivation", "any",
           "keep the current routine going and layer in one stretch goal for the coming week",
           "affirm the consistent pattern explicitly; consistency itself is worth naming"),
        _c("portfolio", "career", "any",
           "start turning one strong project into a portfolio piece — this is a good time to showcase it",
           "suggest a portfolio or showcase opportunity given this student's consistent output"),
        _c("wellbeing", "emotional", "any",
           "keep the balance that's working — enrichment doesn't have to mean more hours, just different ones",
           "no wellbeing concern here; frame any new opportunity as a choice, not an obligation"),
    ],
    "persona_default": [
        _c("concept_review", "academic", "any",
           "review recent course material and reach out with any specific questions",
           "no strong behavioural pattern is on record yet; monitor and re-check after more activity data is available"),
        _c("small_wins", "motivation", "low",
           "pick one small, already-familiar task to build momentum before tackling anything new",
           "consider a light-touch encouragement message rather than a formal intervention"),
        _c("stress_reduction", "emotional", "any",
           "take stock of how the week is going before adding anything new to the plate",
           "a brief, low-key check-in is enough here; no elevated concern is on record"),
        _c("networking", "career", "any",
           "spend a little time this week connecting coursework to a broader goal or interest",
           "mention any relevant project or opportunity in passing during a routine check-in"),
    ],
}


# =========================================================
# PERSONA-SPECIFIC LIBRARY
# =========================================================
# Persona entries here are the ones actually consulted first (see
# get_candidate_pool below) — they take priority over the generic
# backstop, which is what makes persona change the INTERVENTION itself,
# not only its wording. Each persona is defined for every focus id it
# can realistically reach per determine_priority_focus()'s own gating
# (see module docstring): "burnout" only under "Burnout Pattern",
# "enrichment" only under "Consistent Learner"; every other persona
# skips those two keys entirely and relies on the generic backstop for
# them (which they can never actually reach anyway).

_PERSONA_LIBRARY: Dict[str, Dict[str, List[Dict[str, str]]]] = {

    # -----------------------------------------------------
    "Burnout Pattern": {
        "burnout": [
            _c("reduce_workload", "emotional", "high",
               "reduce your workload this week and schedule one real recovery break before your next study session",
               "flag for a workload/wellness check-in; consider a short extension or reduced load"),
            _c("small_wins", "motivation", "medium",
               "set one micro-goal for today — a single short task, nothing bigger",
               "monitor wellbeing; encourage micro-goals rather than a full task list this week"),
            _c("recovery_break", "emotional", "high",
               "take a real recovery break today — not a scroll break, a genuine step away from screens and coursework",
               "recommend deadline flexibility this week given the burnout pattern"),
            _c("stress_reduction", "emotional", "any",
               "protect your sleep routine this week — burnout recovery starts there before anything academic",
               "schedule a short check-in meeting to see how the load is actually feeling, not just how it looks on paper"),
            _c("one_task_planning", "academic", "medium",
               "plan just one task for tomorrow — resist stacking a full list while recovering",
               "scaffold the next assignment into smaller checkpoints rather than one large deadline"),
            _c("habit_building", "motivation", "low",
               "once the immediate pressure eases, celebrate the fact that you kept going — that matters",
               "if the pattern persists after a wellness check-in and reduced load, consider a counselling referral"),
        ],
        "confusion": [
            _c("clarify_concept", "academic", "high",
               "ask for just the one unclear step to be re-explained — don't try to power through the whole topic while also managing burnout",
               "re-explain in a short, low-pressure format; avoid adding volume on top of an already tired student"),
            _c("stress_reduction", "emotional", "any",
               "it's okay that this feels harder right now — confusion on top of burnout is expected, not a sign you're falling behind",
               "note that confusion here may be compounded by fatigue, not just the material itself"),
        ],
        "engagement": [
            _c("recovery_break", "emotional", "high",
               "low engagement right now may just be tiredness — a real break may do more than forcing another session",
               "before pushing more interactive tasks, check whether disengagement here is actually burnout-driven"),
            _c("interactive_task", "academic", "low",
               "when you do have energy, one short interactive task beats a long passive session",
               "offer a single short interactive option, not a full task list, once energy returns"),
        ],
        "inactivity": [
            _c("self_care", "emotional", "high",
               "a stretch of inactivity during burnout is common — restart with one very small task, not a catch-up sprint",
               "reach out with a wellbeing-first message; frame re-engagement as gradual, not a catch-up demand"),
            _c("if_then_plan", "motivation", "medium",
               "pick one specific, small re-entry point (e.g. 'tomorrow morning, one short task') rather than trying to catch up all at once",
               "suggest a gentle, scaffolded re-entry plan rather than a full catch-up expectation"),
        ],
        "persona_default": [
            _c("reduce_workload", "emotional", "high",
               "keep this week's load light — protect the recovery space you've built",
               "continue monitoring wellbeing signals even without a new elevated trigger"),
            _c("habit_building", "motivation", "low",
               "keep the current lighter pace going for now rather than adding anything back too soon",
               "a brief, low-pressure check-in is enough; avoid reintroducing workload too quickly"),
        ],
    },

    # -----------------------------------------------------
    "Passive Watcher": {
        "engagement": [
            _c("interactive_task", "academic", "medium",
               "pick one interactive action today (a short quiz or a discussion post) instead of passive re-reading",
               "prompt with an autonomy-supportive nudge — offer a choice of task rather than a mandate"),
            _c("networking", "career", "low",
               "pick a topic you're curious about and connect it to a project idea, then explore that instead of re-watching material",
               "offer a project-based option — passive review isn't landing; something built might"),
            _c("confidence", "motivation", "low",
               "choose the easiest interactive option first, just to build the habit of doing rather than watching",
               "start with the lowest-effort interactive option available before asking for anything bigger"),
            _c("forum_question", "emotional", "any",
               "post one comment or question, even a small one — the goal is participation, not correctness",
               "invite a specific, low-stakes response rather than open-ended participation"),
        ],
        "confusion": [
            _c("clarify_concept", "academic", "high",
               "ask for the one unclear step directly rather than re-watching the material passively again",
               "offer a worked example — this student tends toward passive re-review rather than asking, so proactively clarifying helps"),
            _c("small_wins", "motivation", "medium",
               "try the practice question actively before deciding it needs another re-watch",
               "encourage an active attempt before more passive material is provided"),
        ],
        "inactivity": [
            _c("resume_activity", "academic", "high",
               "log in and do one small ACTIVE task (not a video) to restart — passive content alone hasn't been sticking",
               "when reaching out, lead with an interactive task rather than more reading or video material"),
            _c("if_then_plan", "motivation", "medium",
               "plan a specific small active task for your next session rather than open-ended review time",
               "suggest a concrete, active re-entry task rather than a general reminder to catch up"),
        ],
        "persona_default": [
            _c("interactive_task", "academic", "any",
               "choose one interactive option today over passive review",
               "offer a choice of task rather than a mandate — autonomy-supportive framing works best here"),
            _c("networking", "career", "low",
               "connect today's material to something you'd want to build, and explore that angle instead",
               "consider a project-based option if a routine check-in reveals continued passivity"),
        ],
    },

    # -----------------------------------------------------
    "Anxiety-Spike Learner": {
        "confusion": [
            _c("stress_reduction", "emotional", "high",
               "it's okay to feel overwhelmed by this — take one slow breath, then ask for just the one unclear step",
               "validate the overwhelm first, then re-explain in a shorter, lower-pressure format"),
            _c("clarify_concept", "academic", "medium",
               "ask for the specific step to be broken down — you don't need to hold the whole topic at once",
               "re-explain with a worked example, paced more slowly than usual"),
            _c("small_wins", "motivation", "low",
               "name one thing you already understand about this topic before tackling the unclear part",
               "open by naming a recent strength before addressing the confusion — reinforce before correcting"),
        ],
        "engagement": [
            _c("stress_reduction", "emotional", "high",
               "shorten your next study session and take a short break partway through to reduce pressure",
               "check in with a reassurance-first message; avoid framing this as a performance warning"),
            _c("confidence", "motivation", "medium",
               "start with a short, low-pressure task to rebuild momentum rather than a long session",
               "offer a low-stakes task and explicitly frame it as no-pressure"),
        ],
        "inactivity": [
            _c("self_care", "emotional", "high",
               "a pause is understandable — restart with one very small, low-pressure task rather than a full catch-up push",
               "reach out gently; avoid language that could read as a warning, given this persona's anxiety pattern"),
            _c("if_then_plan", "motivation", "medium",
               "set one small If-Then plan for restarting, kept deliberately low-stakes",
               "suggest a gentle, specific re-entry point rather than an open-ended catch-up expectation"),
        ],
        "persona_default": [
            _c("stress_reduction", "emotional", "any",
               "shorten your next study session and build in a short break partway through",
               "check in with a reassurance-first tone; avoid framing anything as a performance warning"),
            _c("small_wins", "motivation", "low",
               "name one recent thing that went well before starting anything new",
               "open any check-in by naming a concrete recent strength first"),
        ],
    },

    # -----------------------------------------------------
    "Silent Isolator": {
        "engagement": [
            _c("forum_question", "emotional", "high",
               "ask one question in the course forum or to a classmate this week, even a small one",
               "reach out proactively — this student is unlikely to initiate contact themselves"),
            _c("networking", "career", "medium",
               "consider joining one low-stakes study group or peer discussion this week",
               "suggest a peer-pairing or small-group option rather than expecting self-initiated outreach"),
            _c("interactive_task", "academic", "low",
               "try one interactive task that involves a response from someone else, like a forum reply",
               "flag for proactive educator outreach given the isolation pattern, not just an automated nudge"),
        ],
        "confusion": [
            _c("forum_question", "emotional", "high",
               "post the specific unclear point in the forum rather than working through it alone",
               "proactively reach out with a worked example rather than waiting for this student to ask"),
            _c("clarify_concept", "academic", "medium",
               "ask for the one unclear step — even a short message to someone else counts",
               "re-explain directly; this student is unlikely to request clarification unprompted"),
        ],
        "inactivity": [
            _c("resume_activity", "academic", "high",
               "log in and complete one small piece of coursework today, and consider mentioning it to someone else",
               "reach out directly and personally — inactivity plus isolation makes proactive contact the priority"),
            _c("self_care", "emotional", "medium",
               "if something's going on beyond coursework, a quiet message to a mentor or support contact is a reasonable first step",
               "combine the academic outreach with a genuine, personal check-in rather than an automated reminder"),
        ],
        "persona_default": [
            _c("forum_question", "emotional", "any",
               "ask one small question in the course forum or to a classmate this week",
               "reach out proactively — this student is unlikely to initiate contact themselves"),
            _c("networking", "career", "low",
               "consider one low-stakes social or forum interaction connected to a topic you're interested in",
               "suggest a peer-mentoring pairing given the isolation pattern"),
        ],
    },

    # -----------------------------------------------------
    "Last-Minute Survivor": {
        "inactivity": [
            _c("resume_activity", "academic", "high",
               "log in and complete one small piece of coursework today to break the inactivity streak, then build back up gradually",
               "reach out directly — inactivity is the strongest measured driver of this student's risk"),
            _c("if_then_plan", "motivation", "medium",
               "set a specific If-Then plan (e.g. 'after dinner, review one topic') to space out revision before the deadline",
               "consider a scaffolded reminder or interim checkpoint ahead of the next deadline"),
            _c("habit_building", "motivation", "low",
               "commit to one short, spaced session before the next deadline instead of waiting to cram",
               "highlight the cognitive-load cost of leaving it to the last minute in a brief check-in"),
        ],
        "confusion": [
            _c("clarify_concept", "academic", "high",
               "ask for the unclear step now, before it becomes one more thing to cram at the last minute",
               "re-explain now rather than letting confusion compound closer to the deadline"),
            _c("if_then_plan", "motivation", "medium",
               "set a specific time today to sort out this confusion, rather than deferring it",
               "flag this concept for a scaffolded reminder ahead of the next deadline"),
        ],
        "engagement": [
            _c("if_then_plan", "motivation", "medium",
               "set a specific If-Then micro-habit (e.g. 'after dinner, review one topic') to build steadier engagement between deadlines",
               "prompt with an interactive task tied to an upcoming deadline rather than open-ended review"),
            _c("interactive_task", "academic", "low",
               "do one small interactive task now rather than waiting until closer to the deadline",
               "consider a scaffolded interim checkpoint to encourage earlier engagement"),
        ],
        "persona_default": [
            _c("if_then_plan", "motivation", "any",
               "set a specific If-Then plan tied to a specific deadline, and recommend spaced revision over cramming",
               "consider a scaffolded reminder or interim checkpoint ahead of the next deadline"),
            _c("habit_building", "motivation", "low",
               "name the cognitive-load cost of leaving things to the last minute, then set one small early step",
               "name the cognitive-load cost of last-minute work in a brief, non-judgmental check-in"),
        ],
    },

    # -----------------------------------------------------
    "Consistent Learner": {
        "enrichment": [
            _c("enrichment", "academic", "any",
               "take on an enrichment activity or advanced topic — current performance supports it",
               "consider this student for a peer-mentoring or leadership opportunity"),
            _c("portfolio", "career", "any",
               "turn one strong recent project into a portfolio piece to showcase",
               "suggest a portfolio or showcase opportunity given consistent output"),
            _c("networking", "career", "any",
               "explore a project or internship angle connected to a topic you already do well in",
               "mention a relevant project, internship, or networking opportunity during a routine check-in"),
            _c("habit_building", "motivation", "any",
               "keep the current routine and add one stretch goal for the coming week",
               "affirm the consistent pattern explicitly — consistency itself is worth naming"),
        ],
        "confusion": [
            _c("concept_review", "academic", "medium",
               "revisit the one unclear step — this is a normal gap, not a sign anything's off track",
               "provide a quick worked example; this is routine clarification, not a risk signal"),
            _c("clarify_concept", "academic", "any",
               "ask for the specific unclear point to be broken down before moving to the next topic",
               "re-explain the specific point; frame it as normal rather than a concern"),
        ],
        "engagement": [
            _c("networking", "career", "low",
               "connect the material to a project or interest area to keep things engaging",
               "offer a more advanced or applied option — standard review may simply feel too easy right now"),
            _c("interactive_task", "academic", "low",
               "try a more advanced interactive option instead of standard review",
               "consider offering enrichment-level material rather than remedial framing"),
        ],
        "persona_default": [
            _c("enrichment", "academic", "any",
               "take on an enrichment activity or advanced topic — current performance supports it",
               "consider this student for a peer-mentoring or leadership opportunity"),
            _c("portfolio", "career", "any",
               "consider showcasing recent work — a portfolio piece could be a natural next step",
               "consider this student for a peer-mentoring or leadership opportunity"),
        ],
    },
}


# =========================================================
# CANDIDATE POOL RESOLUTION
# =========================================================

def get_candidate_pool(persona: str, priority_focus: str) -> List[Dict[str, str]]:
    """
    Persona-specific candidates (if any exist for this exact persona +
    focus id) are listed FIRST, so they are preferred over the generic
    backstop at selection time — this is what makes persona change the
    intervention itself, not just its wording (see module docstring).
    Generic candidates are always appended after, both as a fallback
    for personas/foci with no dedicated entries and as extra variety
    for rotation once persona-specific candidates run out.

    Never returns an empty list: `persona_default` in the generic
    library is the absolute last resort if even `priority_focus` is
    unrecognized.
    """
    persona_bucket = _PERSONA_LIBRARY.get(persona, {})
    persona_candidates = persona_bucket.get(priority_focus, [])
    generic_candidates = _GENERIC_LIBRARY.get(priority_focus, [])

    pool = persona_candidates + generic_candidates
    if not pool:
        pool = _GENERIC_LIBRARY.get("persona_default", [])
    return pool


# =========================================================
# SPRINT C — TOPIC-AWARE ACADEMIC ACTIONS (deterministic, rule-based)
# =========================================================
# Context-Aware Academic Recommendation Selection. Problem: even the
# "academic"-intent candidates already in the libraries above are
# study-habit/wellbeing fragments ("ask for the specific step that's
# unclear to be broken down") with no connection to the specific
# academic topic just discussed (e.g. "difference between BFS and
# DFS"). This section adds a SMALL, SEPARATE library of topic-aware
# LEARNING ACTIONS -- solve a practice problem, draw a diagram,
# compare concepts, trace an algorithm, summarize notes, explain
# aloud, create a small example -- matched to the `academic_topic`
# string via FIXED KEYWORD RULES only. No LLM, no embeddings, no
# semantic similarity, no scoring: the first matching rule wins,
# exactly like every other deterministic lookup in this module.
#
# HARD CONSTRAINTS:
# - This is an ADDITIVE candidate source, never a replacement.
#   `get_academic_topic_candidates()` is only ever consulted by
#   `select_recommendation()` (see that function's Sprint C note
#   below), which folds its output into the SAME pool the existing,
#   completely unmodified narrowing algorithm (intent -> risk ->
#   used_action_tags rotation) already consumes.
# - Contributes nothing (`[]`) whenever `academic_topic` is falsy, so
#   any call site that doesn't have one behaves exactly as before this
#   sprint.
# - Requirement #5 of the Sprint C brief: wellbeing/psychological
#   recommendations must still be preserved whenever risk is High.
#   `select_recommendation()` therefore only folds these candidates in
#   when risk is NOT High -- at High risk, the existing persona/
#   generic libraries (which already carry this focus's high-risk-tier
#   psychological candidates) are used completely unmodified.
_TOPIC_KEYWORD_RULES = (
    (
        ("vs", "versus", "difference between", "compare", "comparison"),
        "compare_concepts",
        "compare {topic} side by side and note what's actually similar and what's different",
        "ask the learner to explain when each part of {topic} should be used instead of the other",
    ),
    (
        (
            "algorithm", "bfs", "dfs", "sort", "search", "traversal",
            "recursion", "recursive", "loop", "iteration",
        ),
        "trace_algorithm",
        "trace through {topic} step by step on one small example to see exactly how it behaves",
        "ask the learner to trace {topic} step by step on a small example, out loud",
    ),
    (
        ("graph", "tree", "diagram", "network", "structure", "flow"),
        "draw_diagram",
        "draw a quick diagram of {topic} to make the structure concrete",
        "ask the learner to sketch {topic} and walk you through it",
    ),
    (
        ("notes", "chapter", "unit", "syllabus"),
        "summarize_notes",
        "summarize your notes on {topic} in three or four bullet points",
        "check whether the learner can summarize {topic} in their own words",
    ),
)

# Universal fallbacks, always available regardless of whether a
# keyword rule above matched. Two DIFFERENT fixed actions (not one) so
# `select_recommendation()`'s existing `used_action_tags` rotation has
# a genuinely different option to fall back on once the best-matched
# candidate has already been used, instead of only ever offering one
# topic-aware choice.
_TOPIC_FALLBACK_ACTION = (
    "create_example",
    "create one small original example that illustrates {topic}",
    "ask the learner to invent their own example of {topic} and walk you through it",
)
_TOPIC_SECOND_FALLBACK_ACTION = (
    "explain_aloud",
    "explain {topic} out loud in your own words, as if teaching a friend",
    "ask the learner to explain {topic} out loud, in their own words",
)


def _match_topic_action_types(topic: str) -> List[tuple]:
    """
    Fixed keyword-in-topic substring matching only -- see module note
    above. Returns an ordered list of (tag, student_template,
    educator_template) tuples: the best keyword match first (if any),
    then the two fixed universal fallbacks -- so a caller always has
    more than one option available for rotation, even when no keyword
    matches the topic at all.
    """
    lowered = topic.lower()
    matched: List[tuple] = []
    for keywords, tag, student_template, educator_template in _TOPIC_KEYWORD_RULES:
        if any(keyword in lowered for keyword in keywords):
            matched.append((tag, student_template, educator_template))
            break  # first matching rule wins -- deterministic, no scoring
    matched.append(_TOPIC_FALLBACK_ACTION)
    matched.append(_TOPIC_SECOND_FALLBACK_ACTION)
    return matched


def get_academic_topic_candidates(academic_topic: Optional[str]) -> List[Dict[str, str]]:
    """
    Builds the small, additive topic-aware candidate list described in
    the module note above. Returns `[]` (contributes nothing) when
    `academic_topic` is falsy or blank, so any caller without a real
    topic is completely unaffected.
    """
    if not academic_topic or not academic_topic.strip():
        return []

    topic = academic_topic.strip()
    matches = _match_topic_action_types(topic)
    return [
        _c(
            tag, "academic", "any",
            student_template.format(topic=topic),
            educator_template.format(topic=topic),
        )
        for tag, student_template, educator_template in matches
    ]


# =========================================================
# SELECTION
# =========================================================

def select_recommendation(
    persona: str,
    priority_focus: str,
    risk_level: Optional[str],
    detected_intent: str,
    used_action_tags: Optional[List[str]] = None,
    academic_topic: Optional[str] = None,
) -> Dict[str, str]:
    """
    Deterministic selection of exactly one intervention candidate.

    Inputs match item #2 of the Phase 2.2 brief exactly:
        persona, priority_focus, risk_level, detected_intent,
        used_action_tags

    Output:
        {"student_action": str, "educator_action": str, "tag": str}

    Narrowing order (each stage only relaxes if the stricter stage
    would return nothing, so the pool is NEVER empty going in — see
    get_candidate_pool):
      1. priority_focus narrows the library to this persona's relevant
         intervention family (item #4 of the brief).
      2. detected_intent narrows within that family (item #5). Skipped
         entirely for "general"/unrecognized intent, matching
         intent_classifier.py's own backward-compatibility default.
      3. risk_level biases toward intervention-oriented (high),
         balanced (medium), or growth-oriented (low) candidates
         (item #6). Relaxed if nothing matches this tier.
      4. Repetition avoidance: prefer a candidate whose tag is not in
         used_action_tags. If every remaining candidate has already
         been used, gracefully reuse the first (highest-priority) one
         instead of failing (item #3).

    SPRINT C ADDITION (Context-Aware Academic Recommendation
    Selection): `academic_topic` (new, defaulted to `None` so any
    existing caller that doesn't pass it keeps working unmodified).
    When `detected_intent == "academic"`, `academic_topic` is a real
    string, AND risk is NOT High, the topic-aware learning-action
    candidates from `get_academic_topic_candidates()` are prepended to
    the pool BEFORE the four narrowing stages below run — they are
    simply more entries in the same list, so every existing mechanic
    (intent/risk filtering, used_action_tags rotation, the 4-stage
    relax-and-retry fallback) applies to them exactly as it already
    does to every other candidate. At High risk, this block is skipped
    entirely and the pool is built exactly as it was before this
    sprint, so the persona/generic library's high-risk psychological
    candidates remain the only ones available (brief item #5).
    """
    used = set(used_action_tags or [])
    pool = get_candidate_pool(persona, priority_focus)
    intent = _normalize_intent(detected_intent)
    tier = _risk_tier(risk_level)

    if intent == "academic" and academic_topic and tier != "high":
        pool = get_academic_topic_candidates(academic_topic) + pool

    def _by_intent(cands):
        if intent == "general":
            return cands
        return [c for c in cands if c["intent"] in (intent, "general")]

    def _by_risk(cands):
        return [c for c in cands if c["risk"] in (tier, "any")]

    # Progressively wider stages, each still ordered by the pool's own
    # priority (persona-specific candidates before generic ones). Each
    # stage is only used to look for an UNUSED candidate -- it does not
    # short-circuit selection on its own, so a fully-used narrow stage
    # correctly falls through to a wider one instead of prematurely
    # reusing something. (A tightly-filtered stage can look non-empty
    # even when every entry in it has already been used -- that is not
    # the same as having no unused options left overall.)
    stage1 = _by_risk(_by_intent(pool))  # intent + risk
    stage2 = _by_intent(pool)             # intent only (risk relaxed)
    stage3 = _by_risk(pool)               # risk only (intent relaxed)
    stage4 = pool                         # fully unfiltered

    chosen = None
    for stage in (stage1, stage2, stage3, stage4):
        unused_here = [c for c in stage if c["tag"] not in used]
        if unused_here:
            chosen = unused_here[0]
            break

    if chosen is None:
        # Every candidate in the full pool has already been used --
        # gracefully reuse the highest-priority one from the most
        # specific non-empty stage (item #3 of the brief).
        first_nonempty = next((s for s in (stage1, stage2, stage3, stage4) if s), pool)
        chosen = first_nonempty[0]

    return {
        "tag": chosen["tag"],
        "student_action": chosen["student"],
        "educator_action": chosen["educator"],
    }
