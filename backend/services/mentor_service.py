"""
mentor_service.py (V2)

Sprint 1 refactor of the AI Mentor service.

WHAT CHANGED FROM V1:
- Logic split into context_builder.py, prompt_builder.py, retrieval.py,
  and confidence.py. This file is now an orchestrator, not a monolith.
- The system prompt is now built from clearly labeled sections instead
  of one f-string.
- SHAP explanation and cognitive state are gathered if (and only if)
  upstream services provide them. They are never fabricated — if
  unavailable, the corresponding context fields are None and the
  prompt explicitly tells the LLM not to invent them.
- The mentor now produces a Student Recommendation and an Educator
  Recommendation, in addition to the conversational reply, grounded in
  Risk + SHAP + Persona + Retrieved Knowledge + Student Question
  together (persona guides tone, it does not solely determine content).
- A `confidence` field (High/Medium/Low) is attached, computed only
  from which evidence was actually available (see confidence.py) —
  never asked of or fabricated by the LLM.

WHAT DID NOT CHANGE:
- Public function signature: ask_mentor(user_id: str, question: str)
- Return dict still contains: status, question, mentor_response,
  persona, risk_score, risk_level — so existing FastAPI routes and
  any frontend code consuming this response keep working unmodified.
  New keys (student_recommendation, educator_recommendation,
  confidence, confidence_rationale, shap_available,
  cognitive_state_available) are ADDED, not substituted in place of
  old ones.
- No database schema changes. mentor_history is still written the
  same way, with the same columns.
- No API route changes — this file only changes what happens inside
  the existing function.

INTEGRATION NOTE:
This module assumes it lives alongside context_builder.py,
prompt_builder.py, retrieval.py, and confidence.py (e.g. in the same
`services` package, imported with relative imports). Adjust the import
style below if your project's package layout differs.

TIMING INSTRUMENTATION (added — no behavioural changes):
- Every major stage of ask_mentor() is wrapped in time.perf_counter()
  timers and a formatted timing summary is printed at the end of each
  call (success or error path).
- The per-stage breakdown inside build_student_context() (risk,
  persona, SHAP, cognitive, history) is read from
  `context._timings`, which context_builder.py now populates — see
  that file's own "TIMING INSTRUMENTATION" note.
- Nothing about control flow, return values, prompt content, API
  schema, or model behaviour has changed. This is print-only logging.

SPRINT 3 UPDATE (PROMPT EFFICIENCY OPTIMIZATION):
- ONLY `_build_output_format_instructions()` below was changed, to
  shorten the instructional wording sent to the LLM (paired with the
  section-level optimizations in prompt_builder.py) as part of
  reducing total prompt size from ~9000 to ~2500-3500 characters.
- The three section markers in `_SECTION_MARKERS`, the regex in
  `_parse_structured_response`, the overall 3-part output contract
  (conversational reply / student recommendation / educator
  recommendation), the API response schema, retrieval logic,
  confidence logic, database logic, and every other function in this
  file are UNCHANGED.

SPRINT 5 UPDATE (STRUCTURED-OUTPUT RELIABILITY + PERSONA GROUNDING):
Addresses the AI Mentor Failure Analysis, where every logged scenario
was "Partial Success" because the LLM never reliably emitted the three
structured markers. Three independent, additive fixes, none of which
touch the public API/schema:
  1. `_build_output_format_instructions()` now includes a short
     worked example (few-shot) of the exact format, which measurably
     improves format adherence in small instruction-followers.
  2. The `ollama.chat` call now uses `num_predict=480` (was 250) and
     `temperature=0.5` (was 0.7) — more headroom to finish all three
     sections, and less sampling randomness around the structural
     instruction.
  3. `_parse_structured_response()` now (a) tries a lenient marker
     pattern if the strict one fails, and (b) if the model still omits
     the markers entirely, builds the Student/Educator Recommendation
     deterministically from real context data (persona, SHAP, risk
     reasons) via the new `_fallback_recommendations()` helper, instead
     of returning a static "Not available" string. The conversational
     reply itself is still always the model's own real text.
`_SECTION_MARKERS`, the strict-match contract, the response schema
keys, and `ask_mentor`'s public signature are unchanged.

SPRINT 8 UPDATE (HALLUCINATION SAFETY HARDENING):
`ask_mentor`'s signature and every existing return key are UNCHANGED.
Additive-only changes:
  1. Retrieval now uses `retrieval.TOP_K` instead of a hardcoded `k=5`.
  2. A SIMILARITY THRESHOLD GATE runs right after retrieval: if the
     strongest retrieved chunk's similarity is below
     `retrieval.SIMILARITY_THRESHOLD`, the LLM is never called at all
     -- `ask_mentor` returns the fixed "Insufficient evidence was
     retrieved to answer this question reliably." message directly,
     so a small model is never put in a position to improvise past
     evidence that's too weak to ground an answer (Req: Similarity
     Threshold).
  3. `confidence.assess_confidence()` is now also passed the full
     `retrieved_chunks` list (previously only a boolean), so it can
     compute the new normalized retrieval-confidence score described
     in confidence.py's Sprint 8 update (Req: Confidence Estimation).
  4. `reasoning_layer.classify_retrieval_safety()` is called once per
     request and returned as the new `retrieval_safety_label` key
     (Req: Reasoning Layer -> SAFE / UNCERTAIN / NEEDS_REVIEW).
  5. Every successful (non-gated) answer has a deterministic
     "Sources:" block appended to `mentor_response`, built from
     `prompt_builder.select_chunks_for_prompt()` -- i.e. EXACTLY the
     chunks the model was actually shown, never a larger or
     independently-chosen set (Req: Evidence Citations).
  6. New additive return keys on every path (success, similarity-
     gated, and error): `confidence_score`, `needs_human_review`,
     `retrieval_safety_label`, `sources_used`, `max_similarity`.
No existing key was removed, renamed, or repurposed; no database
schema, route, or prior sprint's logic was touched.

SPRINT 6 UPDATE (EVIDENCE PRIORITIZATION CONSISTENCY):
- `_fallback_recommendations()` now calls prompt_builder.py's new
  `determine_priority_focus()` instead of independently picking
  SHAP/persona evidence. This makes the deterministic fallback reason
  about evidence the SAME way the LLM was instructed to in the prompt
  (see prompt_builder.py's Sprint 6 note), so fallback output doesn't
  disagree with what the prompt told the model to prioritize.
- `_build_output_format_instructions()`'s worked example now uses
  bracketed placeholders instead of naturalistic filled-in prose, to
  stop the model imitating specific wording (e.g. always opening with
  "I hear you") across different learners/turns.
No API/schema change; both are internal-only.

SPRINT 9 UPDATE (BUGFIX: CONFIDENCE FIELD DISAMBIGUATION + RETRIEVAL
GATE HARDENING):
Two additive-only fixes, no API endpoints/routes touched, no existing
key removed or repurposed:
  1. "confidence" (student risk High/Medium/Low) and "confidence_score"
     (mentor response 0..1) were easy to misread as contradictory
     values of one concept. Two new alias keys are added everywhere
     "confidence"/"confidence_score" already appear:
     `student_risk_confidence` and `mentor_response_confidence`. Same
     underlying values, same source of truth (confidence_result) —
     purely a naming fix. See confidence.py's own Sprint update note
     for the full rationale.
  2. The Sprint 8 similarity-threshold gate is now explicitly commented
     as the SOLE authoritative point that blocks the LLM call for weak
     retrieval evidence (see inline comment at the gate itself), with
     the comparison itself pulled into a named boolean
     (`retrieval_evidence_too_weak`) instead of an inline expression,
     so the gate condition can't silently drift out of sync with its
     label across future edits. No threshold value, no gating
     condition, and no behaviour for valid (non-gated) retrievals was
     changed.

SPRINT 1.4 UPDATE (CELEBRATION ROUTING):
Messages classified by intent_classifier.py with `intent == "celebration"`
(e.g. "I passed my exam!", "I finally understood recursion!", "I got
placed!", "I cleared my interview!") must never perform retrieval, never
be subject to the similarity-threshold gate, and never receive
INSUFFICIENT_EVIDENCE_MESSAGE -- they should always get a real,
generated celebratory reply from the LLM via the existing prompt
infrastructure. No new branch was needed: `_RETRIEVAL_INTENTS` and
`_STRICT_GATE_INTENTS` (see the routing block inside `ask_mentor()`)
are allowlists, and "celebration" is simply not added to either --
exactly like "emotional"/"motivation" already aren't. That alone makes
`should_retrieve` and `enforce_strict_gate` both False for celebration,
which means `retrieved_chunks` stays `[]` and the gated-return block
(guarded by `if enforce_strict_gate and retrieval_evidence_too_weak:`)
is never entered, so control falls straight through to conversation-
goal -> system-prompt -> LLM call -> structured parsing, identical to
every other non-gated intent. This sprint only added documentation at
the `_RETRIEVAL_INTENTS`/`_STRICT_GATE_INTENTS` definitions inside
`ask_mentor()`; no executable line changed, no function signature
changed, no key was added to or removed from the return schema, and
`intent_classifier.py`, `prompt_builder.py`, `recommendation_engine.py`,
`recommendation_reasoning.py`, and `reasoning_layer.py` were not
touched.

SPRINT 1.5 UPDATE (POSITIVE LEARNING INTENT):
Celebration messages that also name a specific academic concept (e.g.
"I finally understood recursion!", "I finally understand DBMS.") keep
`detected_intent == "celebration"` exactly as Sprint 1.4 established --
no new intent label was introduced, `_RETRIEVAL_INTENTS` and
`_STRICT_GATE_INTENTS` are untouched, so retrieval and the similarity
gate stay OFF. The only change: `_message_analysis` now also carries
`academic_topic` (see intent_classifier.py's new
`extract_academic_topic()`), and when `detected_intent ==
"celebration"` AND a topic was deterministically found, one small
additive string is appended to `build_system_prompt()`'s own output
(the same technique Sprint 3/5 already use for
`_build_output_format_instructions()`) telling the LLM to celebrate
that specific concept by name. `build_system_prompt()` itself,
`prompt_builder.py`, the API response schema, and every other
existing key/behaviour are unchanged.

SPRINT 7 UPDATE (MULTI-SIGNAL REASONING ARCHITECTURE):
Adds a new `reasoning_layer.py` module and wires it in at two points:
  1. `_build_output_format_instructions()` now also appends a
     persona-varied structural few-shot example.
  2. `ask_mentor()`'s return dict gains two new, additive keys:
     `evidence_profile` and `conversation_plan`.
Persona Memory and Recommendation Diversity (the two pieces of this
sprint that need cross-turn state) are derived ENTIRELY from
`context.chat_history`, which context_builder.py already builds from
the existing `mentor_history` table — no new table, column, query, or
write was added anywhere in this sprint. `ask_mentor`'s signature, the
`mentor_history` schema, every previously-existing return key, and all
prior sprints' logic (context gathering, retrieval, confidence,
parsing, fallback recommendations) are unchanged. See
`reasoning_layer.py`'s module docstring for full detail.
"""

import copy
import re
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import ollama

from config.supabase_client import supabase

from .context_builder import build_student_context, StudentContext
from .prompt_builder import (
    build_system_prompt,
    determine_priority_focus,
    determine_conversation_goal,
    select_chunks_for_prompt,
    detect_comparison_question,
)
from .retrieval import get_retriever, TOP_K, SIMILARITY_THRESHOLD
from .confidence import assess_confidence, CONFIDENCE_THRESHOLD
from .reasoning_layer import (
    get_few_shot_example,
    build_evidence_profile,
    build_conversation_plan,
    classify_retrieval_safety,
    derive_persona_memory,
)
from .prompt_builder import get_persona_actions, get_psychological_framework
from .intent_classifier import classify_intent, analyze_message
from .recommendation_engine import select_recommendation
from .recommendation_reasoning import build_recommendation_rationale

# SPRINT 10 UPDATE (INTENT-AWARE RETRIEVAL ROUTING):
# A new, single-purpose `intent_classifier.classify_intent(question)` is
# called once per request, immediately after context-building and
# BEFORE retrieval. It decides which of four execution paths this
# question takes -- see the routing block inside `ask_mentor()` for the
# full rationale. This is a BRANCH inserted into the existing pipeline,
# not a new one: context-building, confidence.py, reasoning_layer.py,
# prompt_builder.py, response parsing, and the return schema are all
# reused completely unchanged. Only two things vary by intent:
#   1. whether retrieval runs at all, and
#   2. whether a weak/absent match hard-blocks the LLM call or is
#      simply treated as "no sources to show".
# `ask_mentor`'s signature and every existing return key are unchanged;
# the only new key is the additive `detected_intent`.

# SPRINT 7 UPDATE (multi-signal reasoning architecture):
# - `_build_output_format_instructions()` now takes `persona` and
#   appends a persona-varied STRUCTURAL few-shot example (see
#   reasoning_layer.get_few_shot_example) alongside the existing
#   generic bracketed-placeholder example. Still no naturalistic prose
#   for the model to imitate verbatim (Sprint 6's fix is preserved) —
#   only which structural beats are shown varies by persona.
# - `ask_mentor()`'s return dict gains two NEW, ADDITIVE keys only:
#   `evidence_profile` and `conversation_plan`, mirroring what
#   prompt_builder.py told the LLM to anchor on this turn. Every
#   existing key (status, question, mentor_response, persona,
#   risk_score, risk_level, student_recommendation,
#   educator_recommendation, confidence, confidence_rationale,
#   evidence_used, shap_available, cognitive_state_available) is
#   returned exactly as before — no key removed, renamed, or
#   repurposed. Callers that don't know about the two new keys are
#   unaffected.


# =========================================================
# RESPONSE PARSING
# =========================================================
# We ask the LLM for three explicitly labeled sections in one response
# so we don't need 3x the LLM calls. If the model doesn't follow the
# format (it happens), we fail gracefully rather than crash: the whole
# raw response becomes the conversational reply, and the two
# recommendation fields are left as grounded fallback text rather than
# silently empty.

_SECTION_MARKERS = {
    "reply": "###CONVERSATIONAL_REPLY###",
    "student_rec": "###STUDENT_RECOMMENDATION###",
    "educator_rec": "###EDUCATOR_RECOMMENDATION###",
}


# =========================================================
# HALLUCINATION SAFETY (SPRINT 8 ADDITION)
# =========================================================
# Fixed, non-LLM-authored message returned whenever retrieval evidence
# is too weak to safely attempt an answer (Req: Similarity Threshold).
# Kept as a single named constant rather than inlined so it is defined
# exactly once and can be recognized verbatim by downstream tooling or
# tests.
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence was retrieved to answer this question reliably."
)


def _dedupe_preserve_order(items: list) -> list:
    """
    Removes duplicate source names while preserving first-seen order
    (top retrieved chunks are already ranked by relevance, so the
    first occurrence of a given source is also its most-relevant
    occurrence). Plain list utility -- no evidence logic here.
    """
    seen = set()
    deduped = []
    for item in items:
        if item is None or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_sources_block(source_names: list) -> str:
    """
    Deterministically builds the "Sources:" citation block appended to
    every answer that was grounded in retrieved learning material (Req:
    Evidence Citations). This is built in CODE from the exact chunks
    `prompt_builder.select_chunks_for_prompt()` put in front of the
    model -- not parsed out of the LLM's own text -- specifically so a
    small model's tendency to occasionally misname or invent a source
    can never appear in the citation list actually shown to the user.
    """
    if not source_names:
        return ""
    lines = "\n".join(f"- {name}" for name in source_names)
    return f"\n\nSources:\n{lines}"


def _build_output_format_instructions(
    persona: str = "Unknown Persona",
    detected_intent: str = "general",
    is_comparison: bool = False,
) -> str:
    # SPRINT 3 (prompt efficiency): identical contract to before — same
    # three markers, same required content in each section (warm reply;
    # one grounded student action; one grounded educator note; say so
    # plainly if evidence is thin rather than inventing) — just shorter
    # wording. `_SECTION_MARKERS` and `_parse_structured_response`'s
    # parsing regex are untouched, so this remains fully compatible
    # with the existing parsing logic below.
    #
    # SPRINT 5 ADDITION (structured-output reliability): the AI Mentor
    # Failure Analysis showed every scenario landing on "Partial
    # Success" because qwen2.5:3b never reliably emitted the three
    # markers — the parser fallback was firing every time, not just
    # occasionally. A worked, filled-in example (few-shot) is one of
    # the most effective ways to get a small instruction-follower to
    # reproduce an exact format, far more so than a longer description
    # of the format in the abstract. The example below is intentionally
    # generic/neutral (not tied to any persona) so it demonstrates
    # *structure*, not content the model should copy. This is combined
    # with the num_predict/temperature changes and the lenient parser
    # fallback further down, which together address this from three
    # independent angles (prompt, sampling, parsing) rather than
    # relying on any single fix.
    #
    # SPRINT 6 CHANGE: the example's filled-in reply text was originally
    # naturalistic prose ("I hear you — that sounds frustrating...").
    # In practice a 3B model shown one concrete phrase in a few-shot
    # example tends to imitate that *wording*, not just the structure —
    # which would work against the "reduce repeated wording" and
    # "response diversity" goals by giving every learner the same
    # opener. The example below now uses bracketed placeholders even in
    # the worked example, so only the STRUCTURE (marker order, one
    # paragraph per section) is demonstrated, with nothing in
    # naturalistic English left for the model to copy verbatim. It also
    # now explicitly points back at the PRIORITY FOCUS section so the
    # two recommendations are visibly derived from the same anchor as
    # the reply, not independent generic advice.
    # SPRINT 4 ADDITION (renderer-side prompt tweak, item #2 of that
    # sprint's goals): one small, fixed, deterministic instruction line
    # appended ONLY when `detected_intent == "academic"` (default
    # "general" so any existing caller that doesn't pass it keeps
    # getting the exact same instructions as before). Reduces how often
    # the model opens an academic answer with a therapy-style empathy
    # line in the first place; `_strip_academic_empathy_opener()` in
    # `_parse_structured_response()` is the deterministic safety net
    # for whatever still gets through. Does not change the marker
    # structure, the recommendation instructions, or anything else in
    # this function.
    # SPRINT D ADDITION (Academic Answer Excellence): the SPRINT A text
    # below (shared opening two sentences: answer-first, tutor-not-
    # therapist tone) is unchanged. What was previously ONE shared
    # "depth" paragraph is now TWO deterministically-selected variants,
    # chosen by `is_comparison` (computed by `detect_comparison_question`
    # in prompt_builder.py and passed in by the one caller below) --
    # not by the LLM, and not randomly:
    #   - comparison variant: explicit Definition / Key Differences /
    #     Applications / Example structure, for questions like
    #     "difference between BFS and DFS" or "stack vs queue" -- this
    #     directly fixes the reported bug where such questions were
    #     answered as a single-topic explainer that never actually
    #     compared anything.
    #   - normal variant: the same Definition / Explanation / Steps-or-
    #     characteristics / Example structure the prompt already used
    #     (SPRINT A text, verbatim) for academic questions that are NOT
    #     comparisons.
    # Both variants keep every existing hallucination-safety constraint
    # word-for-word (grounded ONLY in Learning material excerpts; never
    # invent an example/fact/difference; say plainly when material is
    # thin instead of padding). Neither variant asks the model to write
    # its own "Sources" section -- that block is still built
    # deterministically in code from `select_chunks_for_prompt()` via
    # `_build_sources_block()`, unchanged from Sprint 8.
    #
    # SPRINT E ADDITION (Academic Answer Excellence, cont'd): three
    # further additive text-only changes, all still inside this same
    # mentor_service.py function -- no prompt SECTION (build_system_
    # prompt's section list in prompt_builder.py) is touched, no
    # retrieval/chunk-selection logic is touched:
    #   1. Goal 1 (answer planning): `_academic_comparison_body` now
    #      explicitly tells the model to give each concept SIMILAR
    #      ATTENTION (not spend most of the answer on the first one
    #      alone) and to move straight from definitions into the
    #      comparison, then applications, then example -- fixing the
    #      reported "explains BFS, then DFS, stops" behaviour and the
    #      "burns the token budget on the first concept" behaviour.
    #      `_academic_normal_body` now also uses explicit labeled
    #      headers (Definition / Explanation / Example) instead of
    #      "natural prose, not necessarily labeled headers" -- this is
    #      the same structural tightening already applied to the
    #      comparison variant in Sprint D, now applied consistently to
    #      the normal variant too, and is what makes Goal 3's
    #      completeness check (below, in `_academic_sections_complete`)
    #      possible to do deterministically rather than guessing.
    #   2. Goal 2 (deterministic checklist): `_academic_checklist`, a
    #      new short, fixed, non-chain-of-thought instruction appended
    #      to BOTH variants, telling the model to silently verify four
    #      fixed conditions before finishing and to fix the answer if
    #      any fail. This is a plain instruction string, identical
    #      every time, with no computation and no visible "thinking" --
    #      it does not ask the model to show its reasoning, only to
    #      use it before finalizing.
    #   3. Goal 4 (never fabricate): both variants now require the
    #      EXACT disclaimer phrase "the retrieved material does not
    #      cover this aspect." (verbatim, matching what the user
    #      requested) whenever the excerpts don't support a given
    #      aspect (time complexity, space complexity, applications, or
    #      an example), instead of the previous looser "say so plainly"
    #      wording. This is worded so the model produces a fixed,
    #      recognizable phrase rather than paraphrasing around a gap.
    _academic_shared_opener = (
        "\n\nThis is an academic question. Begin the reply with the "
        "answer itself (e.g. \"BFS and DFS are both graph traversal "
        "algorithms...\" or \"Here's the difference between...\"). Do "
        "not open with an empathy phrase like \"I understand you're "
        "looking for...\" — answer like a tutor, not a therapist. "
    )

    _academic_comparison_body = (
        "This is a COMPARISON question involving multiple concepts. "
        "Give EACH concept SIMILAR ATTENTION — do not spend most of "
        "the answer explaining only the first one and then stop. "
        "Structure the answer with these explicit labeled sections, "
        "in this exact order, using each label on its own line: "
        "'Definition' (briefly define EACH concept being compared, "
        "giving each roughly equal space), "
        "'Key Differences' (immediately after the definitions — do "
        "not drift back into explaining only one concept first — "
        "compare them directly with a short bullet list covering "
        "whichever of traversal/approach, mechanism, data structure "
        "used, time complexity, space complexity, and use case are "
        "actually supported by the Learning material excerpts above "
        "-- do not invent a dimension the excerpts don't cover), "
        "'Applications' (when each is typically used, if supported), "
        "and finally 'Example' (one simple example ONLY if the "
        "excerpts actually support one). Ground every claim ONLY in "
        "the Learning material excerpts above -- never invent a "
        "difference, a metric, or an example that is not grounded in "
        "those excerpts. If the excerpts don't cover time complexity, "
        "space complexity, applications, or an example for one or "
        "both concepts, write exactly: \"the retrieved material does "
        "not cover this aspect.\" for that part instead of inventing "
        "it."
    )

    _academic_normal_body = (
        "Give a tutor-depth answer, grounded ONLY in the Learning "
        "material excerpts above, using these explicit labeled "
        "sections, each on its own line, in this order: "
        "'Definition' (a concise definition), "
        "'Explanation' (a clear explanation of how it works, plus "
        "whichever of key differences, steps, or characteristics "
        "fits the topic), and finally "
        "'Example' (one simple example ONLY if the Learning material "
        "excerpts actually support one). Never invent an example, a "
        "fact, or a comparison that is not grounded in those "
        "excerpts. If the excerpts are thin or don't cover part of "
        "the question (e.g. time complexity, space complexity, an "
        "application, or an example), write exactly: \"the retrieved "
        "material does not cover this aspect.\" for that part instead "
        "of padding with invented detail."
    )

    # SPRINT E ADDITION (Goal 2): a fixed, deterministic self-check
    # instruction, identical every time regardless of persona/intent/
    # comparison-or-not. This is NOT chain-of-thought -- it does not
    # ask the model to output any reasoning or a visible checklist, it
    # only asks it to confirm these four fixed conditions silently
    # before finalizing the reply and to fix the answer if one fails.
    _academic_checklist = (
        "\n\nBefore finishing, silently verify (do not show this "
        "checklist or any reasoning about it in your reply): "
        "(1) every concept the student asked about was actually "
        "answered, not just the first one; "
        "(2) if this is a comparison question, the comparison was "
        "actually completed, not just two separate definitions; "
        "(3) no required section above was omitted; "
        "(4) every statement is grounded only in the retrieved "
        "material above, with \"the retrieved material does not "
        "cover this aspect.\" used for anything it doesn't support, "
        "instead of invented detail. If any check fails, revise the "
        "answer before responding."
    )

    academic_tone_instruction = (
        (
            _academic_shared_opener
            + (_academic_comparison_body if is_comparison else _academic_normal_body)
            + _academic_checklist
        )
        if detected_intent == "academic"
        else ""
    )

    return (
        "\n\nFormat your ENTIRE reply exactly like this structure (same "
        "three markers, same order, nothing before or after; do not "
        "reuse the bracketed wording literally — write your own):\n"
        f"{_SECTION_MARKERS['reply']}\n"
        "[a warm, specific reply addressing the student's actual "
        "question, shaped by the PRIORITY FOCUS above]\n\n"
        f"{_SECTION_MARKERS['student_rec']}\n"
        "[one concrete action for the student that follows from the "
        "PRIORITY FOCUS and from what the reply just said]\n\n"
        f"{_SECTION_MARKERS['educator_rec']}\n"
        "[one concrete action for the educator that follows from the "
        "same PRIORITY FOCUS]\n\n"
        # SPRINT 7 ADDITION: a second, persona-varied structural
        # example from reasoning_layer.get_few_shot_example(). Still
        # bracketed placeholders only (Sprint 6's fix against wording
        # imitation is preserved) — what varies here is WHICH
        # structural beats are shown for this learner's persona, so
        # the model sees a shape closer to what it should actually
        # produce, not just a generic skeleton.
        "Here is a structural example for a learner with a similar "
        "profile — it shows STRUCTURE only, write your own wording, do "
        "not reuse these bracketed placeholders verbatim:\n"
        f"{get_few_shot_example(persona)}\n\n"
        "Now write your own response in this exact format:\n"
        f"{_SECTION_MARKERS['reply']}\n"
        "<warm conversational reply to the student's question>\n\n"
        f"{_SECTION_MARKERS['student_rec']}\n"
        "<1 short, actionable recommendation FOR THE STUDENT, grounded "
        "in the evidence above; if evidence is thin, say so plainly "
        "instead of inventing one>\n\n"
        f"{_SECTION_MARKERS['educator_rec']}\n"
        "<1 short, actionable recommendation FOR THE EDUCATOR about "
        "this student, same evidence rule>"
        f"{academic_tone_instruction}"
    )


_STRICT_PATTERN = re.compile(
    re.escape(_SECTION_MARKERS["reply"]) + r"(.*?)" +
    re.escape(_SECTION_MARKERS["student_rec"]) + r"(.*?)" +
    re.escape(_SECTION_MARKERS["educator_rec"]) + r"(.*)",
    re.DOTALL,
)

# SPRINT 5 ADDITION: a lenient secondary pattern, tried only if the
# strict one fails. Small models sometimes reproduce the marker text
# with a cosmetic variation (extra/missing whitespace around the
# hashes, or the hashes duplicated/singular) while still clearly
# intending the same three-section structure. This pattern tolerates
# 1-6 leading/trailing '#' characters and surrounding whitespace
# around each marker's core label, but still requires the exact label
# text and the exact section order — it does not "guess" structure
# that isn't there, it only tolerates punctuation noise around markers
# the model clearly meant to produce.
_LENIENT_PATTERN = re.compile(
    r"#{0,6}\s*CONVERSATIONAL_REPLY\s*#{0,6}(.*?)"
    r"#{0,6}\s*STUDENT_RECOMMENDATION\s*#{0,6}(.*?)"
    r"#{0,6}\s*EDUCATOR_RECOMMENDATION\s*#{0,6}(.*)",
    re.DOTALL | re.IGNORECASE,
)


# SPRINT 6 ADDITION: `determine_priority_focus()`'s `reason` field is
# written to be an unambiguous evidence citation inside the LLM prompt
# (e.g. "SHAP top feature='inactivity_days', direction=+risk") — exact
# but not natural-sounding. The deterministic fallback recommendations
# are shown directly to real users, so this maps each focus `id` to a
# short, human-readable grounding phrase instead of reusing the raw
# `reason` string verbatim. This is phrasing only — the underlying
# evidence and decision are identical to what determine_priority_focus
# already computed.
# SPRINT 6 ADDITION: `determine_priority_focus()`'s `reason` field is
# written to be an unambiguous evidence citation inside the LLM prompt
# (e.g. "SHAP top feature='inactivity_days', direction=+risk") — exact
# but not natural-sounding. The deterministic fallback recommendations
# are shown directly to real users, so this maps each focus `id` to a
# short, human-readable grounding phrase instead of reusing the raw
# `reason` string verbatim. This is phrasing only — the underlying
# evidence and decision are identical to what determine_priority_focus
# already computed.
#
# PHASE 2.4 UPDATE: the phrases below are still owned entirely by this
# file (focus SELECTION in determine_priority_focus() is untouched),
# but are now written as noun phrases that complete a full sentence —
# "Your recent activity suggests {human_reason}." — instead of the
# more clinical Sprint 6 wording ("a severe burnout risk pattern"),
# so recommendation_reasoning.py's Phase 2.4 sentence composition
# reads as one coherent thought rather than a label glued onto prose.
# SPRINT 3: each focus_id now maps to a small (2-variant) pool instead
# of one fixed phrase, so the situation sentence built from
# `human_reason` doesn't read identically across every conversation
# that lands on the same focus_id. Variant [0] is the exact original
# phrase, unchanged. Focus SELECTION (determine_priority_focus(),
# recommendation_engine.py) is not touched by this.
_HUMAN_REASON_BY_FOCUS_ID = {
    "enrichment": [
        "steady, consistent progress with no signs of elevated risk",
        "consistently steady progress with nothing indicating elevated risk",
    ],
    "confusion": [
        "some real difficulty following the material clearly",
        "some genuine trouble keeping up with the material",
    ],
    "burnout": [
        "increasing signs of fatigue and strain",
        "a growing pattern of fatigue and strain",
    ],
    "engagement": [
        "engagement that has gradually declined over time",
        "a gradual decline in engagement over time",
    ],
    "inactivity": [
        "a recent stretch of inactivity as the clearest signal right now",
        "a recent stretch of inactivity, which is the clearest signal right now",
    ],
    "persona_default": [
        "the overall learning pattern seen so far as the clearest signal available",
        "the broader learning pattern seen so far, the clearest signal available right now",
    ],
}


def _stable_index(modulo: int, *signal_parts: Any) -> int:
    """
    Deterministic (never random) index in range [0, modulo). Local,
    small copy of the same discipline used in
    recommendation_reasoning.py: identical signals always resolve to
    the identical index, so wording is stable across repeat calls with
    the same context and only varies when the context does.
    """
    if modulo <= 1:
        return 0
    key = "|".join("" if part is None else str(part) for part in signal_parts)
    checksum = sum(ord(ch) for ch in key)
    return checksum % modulo


def _select_variant(pool: Any, salt: str, *signal_parts: Any) -> Optional[str]:
    """Deterministically pick one phrase out of a small hand-written pool."""
    if pool is None or isinstance(pool, str):
        return pool
    if not pool:
        return None
    return pool[_stable_index(len(pool), salt, *signal_parts)]


# =========================================================
# SPRINT 4 — ACTION RENDERING (item #3: wording only, not selection)
# =========================================================
# `focus["student_action"]` / `focus["educator_action"]` are short,
# fixed labels owned entirely by `recommendation_engine.py` (e.g.
# "Take notes.") — WHICH action is chosen is completely untouched
# here. This only builds a richer, more concrete SURFACE string for
# display, by appending one fixed, hand-written "how to do it" clause
# when the action's own wording matches a known generic keyword. No
# topic, number, or fact is invented — every clause below is generic
# enough to apply to any learner, and the action text itself (its
# words, its meaning) is never altered, only extended. If no keyword
# matches, the action is returned completely unchanged, so this is a
# strict enrichment, never a rewrite or a fabrication.
_ACTION_ELABORATION_CLAUSES = (
    (("note",), "jot down the key points as you go so you can review them later"),
    (("session", "study"), "keep it short and focused so it's easier to sustain"),
    (("review", "revisit"), "focus on the parts that felt unclear the first time"),
    (("break", "rest"), "step away fully so you come back with a clearer head"),
    (("ask", "question", "reach out", "office hour"), "so you're not stuck on it alone"),
    (("quiz", "practice", "problem"), "pay close attention to what you got wrong"),
    (("goal", "plan", "schedule"), "write it down somewhere you'll actually see it"),
)


def _elaborate_action(action: str) -> str:
    if not action:
        return action
    stripped = action.strip()
    core = stripped[:-1] if stripped.endswith(".") else stripped
    lowered = core.lower()
    for keywords, clause in _ACTION_ELABORATION_CLAUSES:
        if any(keyword in lowered for keyword in keywords):
            return f"{core} — {clause}."
    return action


# =========================================================
# SPRINT 6 — EDUCATOR RECOMMENDATION QUALITY (renderer-only)
# =========================================================
# Scope: `_render_educator_recommendation` ONLY. `focus["educator_
# action"]` is still the exact short label chosen upstream by
# `recommendation_engine.select_recommendation()` — WHICH action is
# picked, and its priority, are completely untouched here, same as
# Sprint 4's `_elaborate_action` already guaranteed for the student
# voice. This is a SEPARATE elaboration path (`_elaborate_educator_
# action`, not a change to `_elaborate_action` itself), so the
# student-facing renderer and its wording are byte-for-byte unaffected
# by this sprint.
#
# Each pool below is a small set of fixed, hand-written PROFESSIONAL
# case-note templates, matched by the same generic keyword-in-action
# mechanism `_elaborate_action` already uses. No learner fact is
# invented: when a template needs to reference what the action is
# actually about, it reuses the action's OWN text (via `_extract_
# action_subject`, which only strips a small fixed set of leading
# verbs/trailing time phrases already known to appear in these action
# labels — never guesses a topic that wasn't already there). Variant
# selection within a pool is deterministic, using the SAME
# already-computed evidence this file already has on hand for this
# turn (focus_id, risk_level, persona, detected_intent) via the
# existing `_select_variant`/`_stable_index` helpers (Sprint 3) — no
# randomness, no new evidence source.
_EDUCATOR_ACTION_ELABORATION_TEMPLATES = (
    (
        ("mention", "revisit", "review", "recap", "cover"),
        [
            "During the learner's next session, briefly revisit "
            "{action_core}, check whether the concept has been "
            "retained, and reinforce understanding before introducing "
            "more advanced material.",
            "At the start of the next session, take a few minutes to "
            "go back over {action_core} with the learner, confirm it "
            "has stuck, and build on it from there.",
        ],
    ),
    (
        ("reach out", "contact", "check in", "check-in", "follow up"),
        [
            "Schedule a brief check-in to understand the learner's "
            "recent challenges and agree on one achievable short-term "
            "academic goal.",
            "Set aside a short one-on-one conversation to hear how the "
            "learner is doing right now, and leave it with one clear, "
            "achievable next step.",
        ],
    ),
    (
        ("encourage participation", "participation", "involve", "call on"),
        [
            "Create one low-pressure opportunity for the learner to "
            "participate during class so confidence can rebuild "
            "gradually.",
            "Give the learner one small, low-stakes chance to "
            "contribute in class, so participating starts to feel safe "
            "again.",
        ],
    ),
    (
        ("break", "rest", "pace"),
        [
            "Suggest the learner build short, regular breaks into "
            "their study routine, and follow up to see whether that "
            "pacing is actually helping.",
            "Recommend the learner space out their study sessions with "
            "real rest in between, and check back in on whether the "
            "new pace is sustainable.",
        ],
    ),
    (
        ("quiz", "practice", "problem", "exercise"),
        [
            "Assign a short, low-stakes practice set on {action_core}, "
            "then review the results together to see exactly where "
            "support is still needed.",
            "Have the learner work through a few practice questions on "
            "{action_core}, and go over the mistakes together rather "
            "than just the score.",
        ],
    ),
    (
        ("goal", "plan", "schedule"),
        [
            "Work with the learner to set one concrete, achievable "
            "goal around {action_core}, and check back in on progress "
            "at the next session.",
            "Help the learner turn {action_core} into one specific, "
            "written goal, then revisit it together next time.",
        ],
    ),
    (
        ("ask", "question", "office hour"),
        [
            "Invite the learner to bring their specific questions to a "
            "short one-on-one, so they get unstuck instead of "
            "continuing to struggle alone.",
            "Let the learner know they can bring questions on "
            "{action_core} directly to you, rather than working through "
            "the confusion on their own.",
        ],
    ),
    (
        ("note", "notes"),
        [
            "Encourage the learner to keep brief notes during "
            "{action_core}, and check in on whether that habit is "
            "sticking.",
            "Suggest the learner jot down a few key points during "
            "{action_core}, and review them together at the next "
            "check-in.",
        ],
    ),
)

# A small, fixed set of leading verbs and trailing time phrases known
# to appear in `recommendation_engine.py`'s short action labels (e.g.
# "Mention BFS during the next session."). Stripping them isolates
# whatever real topic/subject the action already names — nothing new
# is guessed; if nothing recognizable is left, the caller falls back
# to the action's own full text rather than inventing a subject.
_EDUCATOR_LEADING_VERBS_TO_STRIP = (
    "mention", "revisit", "review", "recap", "cover", "discuss", "reteach", "re-teach",
)
_EDUCATOR_TRAILING_TIME_PHRASES = (
    "during the next session", "during the next class", "in the next session",
    "in the next class", "next session", "next class", "soon", "this week",
)


def _extract_action_subject(core: str) -> Optional[str]:
    """Best-effort, non-inventive extraction of the topic an educator
    action already names (see the module note above). Returns `None`
    — never a guess — when nothing recognizable remains."""
    text = core.strip()
    lowered = text.lower()

    for verb in _EDUCATOR_LEADING_VERBS_TO_STRIP:
        if lowered.startswith(verb + " "):
            text = text[len(verb):].strip()
            lowered = text.lower()
            break

    for phrase in _EDUCATOR_TRAILING_TIME_PHRASES:
        if lowered.endswith(phrase):
            text = text[: len(text) - len(phrase)].strip().rstrip(",")
            lowered = text.lower()
            break

    return text or None


def _looks_sufficiently_detailed(core: str) -> bool:
    """
    Requirement (Sprint 6, item #10): if the action already reads as a
    full, multi-clause professional instruction, leave it unchanged
    rather than layering a template on top of it. Deterministic
    heuristic, no ML: already long enough on its own, or already
    visibly multi-clause (comma-joined) and not trivially short.
    """
    word_count = len(core.split())
    return word_count >= 14 or ("," in core and word_count >= 8)


def _elaborate_educator_action(
    action: str,
    focus_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    persona: Optional[str] = None,
    detected_intent: Optional[str] = None,
) -> str:
    """
    Sprint 6: expands a short educator action label into a fuller,
    professional-register recommendation. Matches the SAME action text
    `recommendation_engine.py` already selected — selection itself is
    completely unchanged (see the module note above). Returns the
    action unchanged when no keyword matches, or when the action
    already looks sufficiently detailed (item #10) — a strict
    enrichment, never a rewrite, never a fact invented about the
    learner.
    """
    if not action:
        return action
    stripped = action.strip()
    core = stripped[:-1] if stripped.endswith(".") else stripped
    lowered = core.lower()

    if _looks_sufficiently_detailed(core):
        return action

    for keywords, templates in _EDUCATOR_ACTION_ELABORATION_TEMPLATES:
        if any(keyword in lowered for keyword in keywords):
            template = _select_variant(
                templates, "educator_action_elaboration",
                focus_id, risk_level, persona, detected_intent,
            )
            if "{action_core}" in template:
                subject = _extract_action_subject(core) or core
                return template.format(action_core=subject)
            return template

    return action


def _combine_lead(memory_clause: Optional[str], sentence: str) -> str:
    """
    PHASE 2.4 ADDITION. Folds a memory clause (already lowercase,
    ending in a comma or semicolon by construction — see
    recommendation_reasoning._memory_clauses()) onto the front of an
    already-capitalized sentence, producing one grammatical sentence
    instead of two disconnected ones. Pure string composition — no new
    evidence, no new claim about the learner.
    """
    if not memory_clause or not sentence:
        return sentence

    continuation = sentence[0].lower() + sentence[1:] if sentence else sentence
    return f"{memory_clause[0].upper()}{memory_clause[1:]} {continuation}"


# =========================================================
# PHASE 2.8 — CONVERSATIONAL FLOW (student renderer ONLY)
# =========================================================
# Phase 2.6 gave each persona a different ORDER of slots. Phase 2.7
# gave each persona different EMOTIONAL CONTENT per slot. Neither
# phase changed how adjacent, already-built sentences are glued
# together -- `_render_student_recommendation` still did a flat
# `" ".join(sentences)`, so every recommendation still reads as a
# checklist (Opening. Observation. Situation. Risk. Recommendation.
# Closing.) even though the sentences behind it now differ by persona.
#
# This section is a pure ASSEMBLY-time change: it decides how the
# exact same, already-built slot sentences (still produced by
# recommendation_reasoning.py -- untouched this phase) are stitched
# together. It never edits a slot's own wording, never adds a new
# slot, never invents a fact, and never changes which slots a
# persona's `conversation_strategy` uses. Two things happen:
#
#   1. SUPPRESS DOUBLE ENCOURAGEMENT: if a persona's strategy places
#      "closing" immediately after "reassurance", the closing sentence
#      is dropped -- reassurance already gave the supportive send-off,
#      so repeating it in different words is the "don't repeat
#      encouragement twice" case called out for this phase. Same
#      "slot present vs. absent" mechanism Phase 2.6 already
#      established for "risk"; nothing new, just applied once more.
#   2. WEAVE ADJACENT SENTENCES: for each adjacent pair of sentences
#      that survives step 1, either (a) the leading sentence already
#      ends in ":" -- every `communication_transition_phrase` is
#      hand-written that way specifically to introduce the next
#      sentence -- in which case the two are joined directly with the
#      next sentence lowercased, or (b) the pair is one of a small,
#      hand-picked set of adjacent-slot combinations that reads
#      naturally as ONE thought (e.g. a risk sentence immediately
#      followed by the recommended action), in which case they are
#      joined with one of the plain connective words called out for
#      this phase ("and", "at the same time,", "even so,", "for this
#      reason,") and the second sentence is lowercased -- otherwise the
#      two stay as separate sentences, exactly as before. At most ONE
#      connector-merge is applied per sentence chunk to avoid a
#      run-on; colon-joins are exempt from that cap since they were
#      already designed to read as one sentence.
#
# No slot's sentence text changes. No new evidence, phrase, or claim
# about the learner is introduced -- only how the SAME sentences are
# ordered/joined changes, and only for the student-facing message.

def _lowercase_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


# Adjacent-slot pairs that read as one flowing thought, and the fixed
# connective word/phrase used to join them. Deliberately small and
# hand-picked (not "merge everything") -- see the module note above.
_SLOT_TRANSITION_CONNECTORS = {
    ("opening", "observation"): "and",
    ("opening", "empathy"): "and",
    ("observation", "situation"): "and",
    ("empathy", "situation"): "and",
    ("situation", "risk"): "at the same time,",
    ("risk", "situation"): "at the same time,",
    ("risk", "action"): "even so,",
    ("situation", "action"): "for this reason,",
    ("reassurance", "transition"): "even so,",
}


def _suppress_redundant_encouragement(resolved):
    """
    Drops a "closing" entry that immediately follows a "reassurance"
    entry -- two encouraging beats in a row with nothing else between
    them reads as the same reassurance said twice. Any other
    arrangement (e.g. reassurance followed by transition, as in the
    Anxiety-Spike strategy) is untouched; closing is only ever dropped
    when it is reassurance's very next neighbor.
    """
    filtered = []
    for slot, text in resolved:
        if slot == "closing" and filtered and filtered[-1][0] == "reassurance":
            continue
        filtered.append((slot, text))
    return filtered


def _weave_slot_sentences(strategy: List[str], slot_text: Dict[str, Optional[str]]) -> str:
    """
    Assembles this persona's already-built slot sentences (unchanged
    content) into flowing prose instead of a flat, space-joined list.
    See the Phase 2.8 module note above for the full rationale.
    """
    resolved = [(slot, slot_text[slot]) for slot in strategy if slot_text.get(slot)]
    resolved = _suppress_redundant_encouragement(resolved)
    if not resolved:
        return ""

    chunks: List[str] = []
    current_slot, current_text = resolved[0]
    merged_this_chunk = False

    for slot, text in resolved[1:]:
        if current_text.rstrip().endswith(":"):
            # The leading sentence was hand-written to introduce the
            # next one -- join directly, no connector needed.
            current_text = f"{current_text.rstrip()} {_lowercase_first(text)}"
            current_slot = slot
            continue

        connector = None if merged_this_chunk else _SLOT_TRANSITION_CONNECTORS.get((current_slot, slot))
        if connector:
            base = current_text.rstrip()
            if base.endswith((".", "!", "?")):
                base = base[:-1]
            current_text = f"{base}, {connector} {_lowercase_first(text)}"
            current_slot = slot
            merged_this_chunk = True
        else:
            chunks.append(current_text)
            current_slot, current_text = slot, text
            merged_this_chunk = False

    chunks.append(current_text)
    return " ".join(chunks)


def _render_student_recommendation(
    focus: Dict[str, str],
    human_reason: str,
    detected_intent: str = "general",
    used_action_tags: Optional[List[str]] = None,
    rationale: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    """
    PHASE 2.1 ADDITION, PHASE 2.3 UPDATE, PHASE 2.4 UPDATE, PHASE 2.5
    UPDATE, PHASE 2.6 UPDATE, PHASE 2.7 UPDATE — owns the STUDENT voice
    only: second person ("you").

    Selection logic is still UNCHANGED — `focus["student_action"]`
    still comes from `determine_priority_focus()` + (as of Phase 2.2)
    `recommendation_engine.select_recommendation()`; this function only
    controls how that already-chosen action is worded and, as of
    Phase 2.6, in what order/emphasis it and the surrounding sentences
    are presented.

    PHASE 2.5: introduced persona-voiced sentences — `persona_opening`,
    `persona_encouragement_template`, `persona_closing` (see
    `recommendation_reasoning.py`'s Phase 2.5 note) — as well as the
    Priority-Focus-driven content sentences (`opening_observation`,
    `situation_student`, `risk_sentence_student`). Phase 2.5 still
    assembled all of these in ONE fixed order for every persona.

    PHASE 2.6 CHANGE: each of the (up to) six possible sentences —
    opening / observation / situation / risk / action / closing — is
    still built with EXACTLY the same content and wording as Phase
    2.5. What changes is that they are now placed into a `slot_text`
    lookup and assembled according to `rationale["conversation_plan"
    ... conversation_strategy]` (built once in
    `recommendation_reasoning.get_conversation_strategy(persona)`), an
    ordered list of which slots this persona's mentoring style uses and
    in what sequence. This is a pure select/reorder step — no new
    sentence, no new evidence, no change to `determine_priority_focus`
    or `recommendation_engine.select_recommendation` — so different
    personas now walk genuinely different conversational strategies
    (e.g. Burnout Pattern never states the risk sentence at all; Passive
    Watcher states it FIRST) instead of the same six-step script with
    different words. `rationale` is still optional and defaults to
    `None`, so any existing caller that doesn't pass it keeps getting
    the original, simplest template (backward compatible); a rationale
    dict built by an older version of `recommendation_reasoning.py`
    (without `conversation_strategy`) still works, since this function
    falls back to the original fixed six-slot order in that case.

    PHASE 2.7 CHANGE: `slot_text` gains three more entries --
    "empathy", "transition", "reassurance" -- sourced from
    `recommendation_reasoning.get_communication_profile(persona)` via
    `rationale["communication_empathy_phrase"]` etc. These are NEW,
    persona-unique sentences (never reused/reworded from
    persona_opening/closing/encouragement_template), inserted purely
    for emotional register (empathy, reassurance, a softer/brisker
    hand-off into the action) -- they carry no evidence and never
    appear unless this persona's `conversation_strategy` names that
    slot, so a persona described as "lower empathy" (e.g. Passive
    Watcher) simply never includes the "empathy"/"reassurance" slots
    and its output is unaffected in tone. The action string itself,
    `human_reason`, and every evidence-bearing sentence are completely
    untouched by this phase.

    PHASE 2.8 CHANGE (this phase, renderer-only): the final assembly
    step no longer does a flat `" ".join(sentences)`. It now calls
    `_weave_slot_sentences()` (defined just above this function), which
    (a) drops a "closing" sentence that immediately repeats a
    "reassurance" sentence, and (b) joins a small, fixed set of
    adjacent-slot pairs into one flowing sentence with a plain
    connective word ("and", "at the same time,", "even so,", "for this
    reason,") instead of a hard sentence break, plus always flows a
    sentence directly into whatever follows a ":"-ending transition
    phrase. No slot's own wording changes, no slot is added, no new
    evidence is introduced, and `conversation_strategy` (Phase 2.6) and
    every slot's content (Phase 2.4/2.5/2.7) are read exactly as
    before -- only how the resulting sentences are glued together is
    different. See the Phase 2.8 module note above `_lowercase_first`.
    """
    action = focus["student_action"]

    if not rationale:
        return f"Today, you could {_elaborate_action(action)} — a supportive next step, given {human_reason}."

    # PHASE 2.6 CHANGE: sentences are no longer assembled in one fixed
    # Opening -> Observation -> Interpretation -> Risk -> Recommendation
    # -> Closing order for every persona. Each of the six possible
    # sentences is first built EXACTLY as before (same content, same
    # evidence, same Phase 2.4/2.5 wording) into a `slot_text` lookup;
    # a persona-specific CONVERSATION STRATEGY (an ordered list of slot
    # names from `recommendation_reasoning.get_conversation_strategy`,
    # via `rationale["conversation_strategy"]`) then decides which of
    # these already-built sentences are used and in what order. No
    # sentence's wording changes and no new evidence is introduced here
    # — this only changes selection/ordering of sentences that were
    # already going to be produced.

    # Interpretation/situation — one coherent sentence, with any real
    # prior support folded naturally onto the front of it. Length is
    # capped per-persona upstream (Phase 2.5's `max_extra_clauses`), so
    # this is also where sentence length varies by persona.
    situation = rationale.get("situation_student") or (
        f"Your recent activity suggests {human_reason}."
    )
    situation = _combine_lead(rationale.get("memory_clause_student"), situation)

    # Recommendation — the single action already selected upstream,
    # wrapped in this persona's own encouragement phrasing/word choice
    # (Phase 2.5), not one generic sentence for every persona.
    encouragement_template = rationale.get("persona_encouragement_template") or (
        "Given all this, choosing to {action} could really help right now."
    )

    # SPRINT B ADDITION (Recommendation Relevance): fold the new,
    # optional academic-context lead-in ("after reviewing BFS and
    # DFS,") onto the front of the SAME action sentence, using the
    # exact same `_combine_lead` helper already used above for the
    # memory clause. `action` itself, `focus["student_action"]", and
    # `_elaborate_action()` are completely unchanged -- this only adds
    # a lead-in clause in front of the wording they already produced.
    # Resolves to a no-op (returns `action_sentence` unchanged) for
    # every non-academic intent and for any academic question without
    # a deterministically extracted topic -- see
    # recommendation_reasoning._academic_context_clauses().
    action_sentence = encouragement_template.format(action=_elaborate_action(action))
    action_sentence = _combine_lead(rationale.get("academic_context_student"), action_sentence)

    slot_text: Dict[str, Optional[str]] = {
        "opening": rationale.get("persona_opening"),
        "observation": rationale.get("opening_observation"),
        "situation": situation,
        "risk": rationale.get("risk_sentence_student"),
        "action": action_sentence,
        "closing": rationale.get("persona_closing"),

        # PHASE 2.7 ADDITION: three new, persona-unique phrases from
        # `recommendation_reasoning.get_communication_profile()`. Same
        # "resolve to None -> automatically skipped" behaviour as every
        # other slot here, so a rationale built by an older version of
        # recommendation_reasoning.py (without these keys) still works
        # unmodified.
        "empathy": rationale.get("communication_empathy_phrase"),
        "transition": rationale.get("communication_transition_phrase"),
        "reassurance": rationale.get("communication_reassurance_phrase"),
    }

    # PHASE 2.6: the ordered list of which slots this persona's
    # mentoring strategy uses. Falls back to the original fixed
    # six-slot order (identical to pre-Phase-2.6 behaviour) if the
    # rationale doesn't carry a strategy — e.g. an older caller that
    # built its own rationale dict without this new, additive key.
    strategy = rationale.get("conversation_strategy") or [
        "opening", "observation", "situation", "risk", "action", "closing",
    ]

    # PHASE 2.8 CHANGE: previously `" ".join(sentences)` -- every
    # slot's sentence, in strategy order, glued together with nothing
    # but a space. That is what made the output read as a checklist
    # (Opening. Observation. Situation. Risk. Recommendation. Closing.)
    # no matter how much Phase 2.6/2.7 varied the slots or their
    # wording. Same slots, same order, same content -- weaving only
    # changes how adjacent sentences are JOINED: dropping a "closing"
    # that immediately repeats a "reassurance", and gluing a handful
    # of adjacent-slot pairs into one flowing sentence with a plain
    # connective word instead of a hard sentence break. See the
    # Phase 2.8 module note above `_lowercase_first` for the full
    # rationale.
    return _weave_slot_sentences(strategy, slot_text)


def _render_educator_recommendation(
    focus: Dict[str, str],
    human_reason: str,
    detected_intent: str = "general",
    used_action_tags: Optional[List[str]] = None,
    rationale: Optional[Dict[str, Optional[str]]] = None,
    risk_level: Optional[str] = None,
    persona: Optional[str] = None,
) -> str:
    """
    PHASE 2.1 ADDITION, PHASE 2.3 UPDATE, PHASE 2.4 UPDATE — the
    educator-side counterpart to `_render_student_recommendation`.
    Third person only ("the learner"/"this student"), professional
    case-note register. Summarizes the learner's situation first, then
    recommends the intervention (item #4 of the Phase 2.3 brief).

    PHASE 2.4 CHANGE: the situation summary is now the single composed
    `situation_educator` sentence (with any real memory folded onto the
    front of it) plus a plain-language risk sentence, instead of one
    comma-joined `human_reason` fragment. Deliberately does NOT add a
    "closing_line" — item #8 of the Phase 2.4 brief asks the educator
    voice to stay a professional case note, not a pep talk, so student-
    facing encouragement is intentionally not mirrored here.

    Selection logic is still UNCHANGED — see
    `_render_student_recommendation` docstring. `rationale` is optional
    and defaults to `None` for the same backward-compatibility reason.

    SPRINT 6 ADDITION (Educator Recommendation Quality, renderer-only):
    the action is now expanded via `_elaborate_educator_action()`
    instead of the shared `_elaborate_action()` — a dedicated,
    educator-only elaboration path (see that function's module note),
    so `_render_student_recommendation` is completely unaffected.
    `risk_level` and `persona` are new, optional, defaulted-to-`None`
    parameters (backward compatible with any existing caller) used
    ONLY to deterministically pick among a template's hand-written
    wording variants — never to change which action is selected or
    its priority.
    """
    action = focus["educator_action"]
    elaborated_action = _elaborate_educator_action(
        action, focus.get("id"), risk_level, persona, detected_intent
    )

    if not rationale:
        return f"Case note: this learner is showing {human_reason}. Recommended follow-up — {elaborated_action}."

    # SPRINT B ADDITION (Recommendation Relevance): same fold as the
    # student renderer above, applied to the educator's follow-up
    # action instead of the situation summary. `action`,
    # `focus["educator_action"]`, and `_elaborate_educator_action()`
    # are completely unchanged -- this only adds an optional lead-in
    # clause ("during the next discussion on BFS and DFS,") in front of
    # the wording they already produced. No-op for every non-academic
    # intent and for any academic question without a deterministically
    # extracted topic.
    elaborated_action = _combine_lead(rationale.get("academic_context_educator"), elaborated_action)

    situation = rationale.get("situation_educator") or (
        f"This learner's recent activity suggests {human_reason}."
    )
    situation = _combine_lead(rationale.get("memory_clause_educator"), situation)

    summary_parts = [situation]
    risk_sentence = rationale.get("risk_sentence_educator")
    if risk_sentence:
        summary_parts.append(risk_sentence)

    summary = " ".join(summary_parts)

    return f"{summary}\n\nRecommended follow-up — {elaborated_action}."


def _fallback_recommendations(
    context: Optional["StudentContext"],
    detected_intent: str = "general",
    question: str = "",
    conversation_goal: Optional[str] = None,
    short_term_context_relevant: bool = True,
    academic_topic: Optional[str] = None,
) -> Dict[str, str]:
    """
    SPRINT 5 ADDITION, updated in SPRINT 6 / PHASE 2.1 / PHASE 2.2 / PHASE 2.3.

    Builds a Student Recommendation and Educator Recommendation
    deterministically from data ALREADY in `context`, for use only
    when the LLM fails to produce the structured markers (both the
    strict and lenient parse attempts miss).

    PHASE 2.2 (unchanged this phase): `determine_priority_focus()`
    picks the evidence focus; `recommendation_engine.select_recommendation()`
    picks the concrete intervention out of a persona/focus/intent/risk-
    aware candidate library, rotated away from `used_action_tags`.
    Neither of those is touched here.

    PHASE 2.3 ADDITION: previously the SAME `focus`/`human_reason` pair
    was handed straight to the renderers, which wrapped the selected
    action in one fixed template. Now, `recommendation_reasoning.
    build_recommendation_rationale()` first turns that same evidence —
    plus the student's actual `question`, cognitive state, SHAP
    explanation, and conversation memory — into a small set of natural-
    language fragments (never raw scores), which the renderers use to
    produce a recommendation that references what the learner actually
    asked and, when real history exists, acknowledges it. `question`
    defaults to `""` so any existing caller that doesn't pass it keeps
    working — the rationale then simply omits the question reference,
    same as when there is no question.

    PHASE 3.3 ADDITION (Adaptive Conversation Planning): `conversation_goal`
    (new, defaulted to `None` so any existing caller that doesn't pass
    it keeps working exactly as before) is threaded straight through to
    `build_recommendation_rationale()`, which uses it ONLY to reorder
    (never change the content of) the persona's existing
    conversation-strategy slots -- see recommendation_reasoning.py's
    Phase 3.3 note. `determine_priority_focus()` and
    `select_recommendation()` below are completely unaffected.

    SPRINT 2 ADDITION: `short_term_context_relevant` (defaulted to
    `True` so any existing caller that doesn't pass it keeps working
    exactly as before) is threaded straight through to
    `build_recommendation_rationale()`, which uses it ONLY to suppress
    the memory-clause callback when `False` -- see
    recommendation_reasoning.py's Sprint 2 note. `determine_priority_
    focus()` and `select_recommendation()` below are unaffected.

    SPRINT B ADDITION (Recommendation Relevance): `academic_topic`
    (defaulted to `None`, same backward-compatibility reasoning) is
    threaded straight through to `build_recommendation_rationale()`,
    which uses it ONLY to add an optional lead-in clause referencing
    the academic concept just discussed -- see that function's Sprint B
    note and `recommendation_reasoning._academic_context_clauses()`.
    `determine_priority_focus()`, `select_recommendation()`, and every
    other field built below are completely unaffected.
    """

    if context is None:
        not_available = (
            "Not available — the mentor's response did not include a "
            "structured recommendation this time."
        )
        return {
            "student_recommendation": not_available,
            "educator_recommendation": not_available,
        }

    focus = determine_priority_focus(context)

    memory = derive_persona_memory(getattr(context, "chat_history", "") or "")
    used_action_tags = memory["used_action_tags"]

    # SPRINT 3: same evidence-to-phrase mapping as before, now picking
    # deterministically among 2 hand-written phrases per focus_id using
    # signals already available here (persona, focus_id, risk_level,
    # detected_intent, and how many actions this learner has already
    # been given) instead of always returning the pool's only phrase.
    human_reason = _select_variant(
        _HUMAN_REASON_BY_FOCUS_ID.get(focus["id"], ["the available evidence"]),
        "human_reason",
        context.persona,
        focus["id"],
        context.risk_level,
        detected_intent,
        len(used_action_tags),
    )

    # PHASE 2.2 (unchanged): selection only, no wording decisions here.
    # SPRINT C ADDITION (Context-Aware Academic Recommendation
    # Selection): `academic_topic` is passed straight through -- this
    # function already receives it (Sprint B). See
    # `recommendation_engine.select_recommendation()`'s own Sprint C
    # note for what it does with it (nothing changes here beyond
    # passing the value along).
    selection = select_recommendation(
        persona=context.persona,
        priority_focus=focus["id"],
        risk_level=context.risk_level,
        detected_intent=detected_intent,
        used_action_tags=used_action_tags,
        academic_topic=academic_topic,
    )
    focus_for_render = dict(focus)
    focus_for_render["student_action"] = selection["student_action"]
    focus_for_render["educator_action"] = selection["educator_action"]

    # PHASE 2.3 ADDITION, PHASE 2.5 UPDATE: natural-language rationale,
    # internal only — never returned to an API consumer, only used to
    # shape the two rendered strings below. `persona` (Phase 2.5) is
    # `context.persona` — the SAME value already used to pick
    # `focus["student_action"]`/`focus["educator_action"]` above; it is
    # not a second, independently-derived persona.
    rationale = build_recommendation_rationale(
        human_reason=human_reason,
        detected_intent=detected_intent,
        risk_level=context.risk_level,
        focus_id=focus["id"],
        cognitive_state=(context.cognitive_state if getattr(context, "cognitive_state_available", False) else None),
        shap_explanation=(context.shap_explanation if getattr(context, "shap_available", False) else None),
        question=question,
        memory=memory,
        persona=context.persona,
        conversation_goal=conversation_goal,
        short_term_context_relevant=short_term_context_relevant,
        academic_topic=academic_topic,
    )

    student_recommendation = _render_student_recommendation(
        focus_for_render, human_reason, detected_intent, used_action_tags, rationale
    )
    educator_recommendation = _render_educator_recommendation(
        focus_for_render, human_reason, detected_intent, used_action_tags, rationale,
        risk_level=context.risk_level, persona=context.persona,
    )

    return {
        "student_recommendation": student_recommendation,
        "educator_recommendation": educator_recommendation,
    }


# =========================================================
# SPRINT 2 ADDITION -- SHORT-TERM CONTEXT RELEVANCE GATING
# =========================================================
# Deterministic, side-effect-free helper: decides whether THIS turn's
# short-term chat_history should be carried forward at all (into the
# system prompt and into the recommendation rationale's memory
# clauses), or whether it should be treated as not relevant for this
# turn. Content-focused intents (academic/career) are treated as fresh
# lookups -- e.g. an exam question about BFS should never have an
# unrelated earlier turn's chat history bleeding into its prompt or
# its "since you've already been working on..." callback. Emotional/
# motivation/celebration/general-support intents keep the short-term
# context, since continuity is exactly what makes those turns feel
# like an ongoing conversation rather than a cold restart.
#
# This does not change `detected_intent` itself, does not touch
# retrieval routing/the similarity gate/confidence/persona selection,
# and does not delete any data -- it only decides, for this turn,
# whether short-term chat_history gets threaded into the prompt and
# rationale below.
_SHORT_TERM_CONTEXT_IRRELEVANT_INTENTS = {"academic", "career"}


def _short_term_context_is_relevant(detected_intent: str) -> bool:
    return detected_intent not in _SHORT_TERM_CONTEXT_IRRELEVANT_INTENTS


# =========================================================
# SPRINT 4 — RENDERER-LAYER SAFETY NETS (deterministic, no LLM)
# =========================================================
# Two renderer-only guards, both applied purely to text already
# produced above (the LLM's raw output / the deterministic fallback
# templates). Neither touches `determine_priority_focus()`,
# `recommendation_engine.select_recommendation()`, or which action/
# focus was chosen -- only what the final strings look like.

# GOAL 1: the few-shot prompt in `_build_output_format_instructions()`
# necessarily shows the model bracketed placeholder syntax as part of
# demonstrating STRUCTURE (Sprint 6's fix against wording imitation).
# A small instruction-follower model occasionally echoes that
# placeholder text verbatim instead of filling it in. This is a fixed,
# deterministic check for that literal leakage -- it never flags real
# recommendation prose, which never legitimately contains "<...>" or
# these exact instructional phrases.
_PLACEHOLDER_MARKERS = (
    "for the student",
    "for the educator",
    "short, actionable recommendation",
    "grounded in the evidence above",
    "same evidence rule",
)


def _looks_like_placeholder(text: Optional[str]) -> bool:
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if "<" in stripped and ">" in stripped:
        return True
    lowered = stripped.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


# GOAL 2: academic answers should open with the answer, not a
# therapy-style empathy opener. This only ever fires for
# `detected_intent == "academic"` -- every other intent's opener is
# completely untouched. Fixed, hand-written pattern list (same
# deterministic, no-fabrication discipline as the rest of this
# module); strips at most the one leading clause/sentence it
# recognizes and leaves everything else -- including the rest of the
# same sentence -- exactly as the model wrote it.
_ACADEMIC_EMPATHY_OPENER = re.compile(
    r"^\s*("
    r"i understand[^.!?]*[.!?]\s*|"
    r"i hear you[^.!?]*[.!?]\s*|"
    r"i (can (see|imagine)|know)[^.!?]*[.!?]\s*|"
    r"i'?m (sorry|glad)[^.!?]*[.!?]\s*|"
    r"that (sounds|must)[^.!?]*[.!?]\s*"
    r")",
    re.IGNORECASE,
)


def _strip_academic_empathy_opener(reply: str, detected_intent: str) -> str:
    if detected_intent != "academic" or not reply:
        return reply
    without_opener = _ACADEMIC_EMPATHY_OPENER.sub("", reply, count=1).lstrip()
    if not without_opener or without_opener == reply:
        return reply
    if without_opener[0].islower():
        without_opener = without_opener[0].upper() + without_opener[1:]
    return without_opener


def _parse_structured_response(
    raw_text: str,
    context: Optional["StudentContext"] = None,
    detected_intent: str = "general",
    question: str = "",
    conversation_goal: Optional[str] = None,
    short_term_context_relevant: bool = True,
    academic_topic: Optional[str] = None,
) -> Dict[str, str]:
    """
    Splits the LLM's raw output into the three labeled sections.

    Tries the strict marker pattern first, then a lenient variant that
    tolerates cosmetic marker noise, before falling back. On fallback,
    the conversational reply is still the model's real raw text (never
    discarded), but the two recommendation fields are now built by
    `_fallback_recommendations` from real context data instead of a
    static "Not available" string — see that function's docstring.

    `context` is optional and keyword-only in spirit (positional here
    only for call-site brevity) so existing/external callers invoking
    this with just `raw_text` keep working unmodified; they simply get
    the original honest "not available" fallback text, same as before.

    PHASE 2.1 ADDITION: `detected_intent` (default "general" so any
    existing caller that doesn't pass it keeps working unmodified) is
    threaded straight through to `_fallback_recommendations` on both
    fallback paths below.

    PHASE 2.3 ADDITION: `question` (also defaulted, same backward-
    compatibility reasoning) is threaded through as well, so the
    fallback recommendations can naturally reference what the student
    actually asked (see recommendation_reasoning.py).

    PHASE 3.3 ADDITION: `conversation_goal` (also defaulted to `None`,
    same backward-compatibility reasoning) is threaded through to
    `_fallback_recommendations` on both fallback paths below, so the
    same conversational goal driving the LLM's own prompt this turn is
    also available to the deterministic fallback's slot ordering.

    SPRINT 2 ADDITION: `short_term_context_relevant` (defaulted to
    `True`, same backward-compatibility reasoning) is threaded through
    to `_fallback_recommendations` on both fallback paths below, so a
    turn whose short-term chat history was judged not relevant (see
    `_short_term_context_is_relevant()`) doesn't get a memory-clause
    callback added into its fallback recommendation either.

    SPRINT B ADDITION (Recommendation Relevance): `academic_topic`
    (defaulted to `None`, same backward-compatibility reasoning) is
    threaded through to `_fallback_recommendations` on all three
    fallback paths below, so a deterministically-built recommendation
    for an academic question can reference the concept just discussed
    -- see `_fallback_recommendations`'s own Sprint B note.
    """

    match = _STRICT_PATTERN.search(raw_text)
    if match:
        reply, student_rec, educator_rec = match.groups()
        reply = _strip_academic_empathy_opener(reply.strip(), detected_intent)
        student_rec, educator_rec = student_rec.strip(), educator_rec.strip()
        if _looks_like_placeholder(student_rec) or _looks_like_placeholder(educator_rec):
            fallback = _fallback_recommendations(context, detected_intent, question, conversation_goal, short_term_context_relevant, academic_topic)
            if _looks_like_placeholder(student_rec):
                student_rec = fallback["student_recommendation"]
            if _looks_like_placeholder(educator_rec):
                educator_rec = fallback["educator_recommendation"]
        return {
            "mentor_response": reply,
            "student_recommendation": student_rec,
            "educator_recommendation": educator_rec,
            # SPRINT 11 ADDITION (Confidence Presentation Calibration):
            # internal-only signal, never surfaced in the API response —
            # see _refine_response_confidence()'s docstring. Purely
            # additive; every existing key above is unchanged, and this
            # extra key is harmless for any caller that ignores it.
            "_parse_method": "strict",
        }

    lenient_match = _LENIENT_PATTERN.search(raw_text)
    if lenient_match:
        reply, student_rec, educator_rec = lenient_match.groups()
        reply = _strip_academic_empathy_opener(reply.strip(), detected_intent)
        student_rec, educator_rec = student_rec.strip(), educator_rec.strip()
        if _looks_like_placeholder(student_rec) or _looks_like_placeholder(educator_rec):
            fallback = _fallback_recommendations(context, detected_intent, question, conversation_goal, short_term_context_relevant, academic_topic)
            if _looks_like_placeholder(student_rec):
                student_rec = fallback["student_recommendation"]
            if _looks_like_placeholder(educator_rec):
                educator_rec = fallback["educator_recommendation"]
        return {
            "mentor_response": reply,
            "student_recommendation": student_rec,
            "educator_recommendation": educator_rec,
            "_parse_method": "lenient",
        }

    # Neither pattern matched at all. Don't crash, don't invent the
    # conversational reply — treat the whole raw text as the reply
    # (unchanged behaviour), but ground the two recommendations in
    # real evidence instead of a static placeholder.
    fallback = _fallback_recommendations(context, detected_intent, question, conversation_goal, short_term_context_relevant, academic_topic)
    return {
        "mentor_response": _strip_academic_empathy_opener(raw_text.strip(), detected_intent),
        "student_recommendation": fallback["student_recommendation"],
        "educator_recommendation": fallback["educator_recommendation"],
        "_parse_method": "fallback",
    }


# =========================================================
# SPRINT 11 ADDITION (CONFIDENCE PRESENTATION CALIBRATION)
# =========================================================
# `confidence.py` is NOT modified by this sprint. `assess_confidence()`
# computes `normalized_score` from which EVIDENCE CATEGORIES exist for
# this student at all (SHAP present? persona present? was retrieval
# attempted?) — a coarse, request-shape signal. It has no way to know
# HOW WELL this specific turn's retrieval actually went: whether the
# best chunk cleared the similarity threshold with real margin, whether
# any source actually made it into the "Sources:" block, or whether the
# LLM's own output matched the structured 3-marker contract or had to
# fall back to a deterministically-built recommendation. Those signals
# already exist elsewhere in this file (the similarity gate, the
# citation step, `_parse_structured_response`'s new `_parse_method`
# tag) — this function is the ONLY place that combines them into a
# presentation-layer refinement of the response-grounding axis
# (`confidence_score` / `mentor_response_confidence` / the resulting
# `needs_human_review`). It never touches `confidence_result` itself,
# never calls into confidence.py again, and never touches the
# student-risk axis (`confidence` / `student_risk_confidence`), which
# stays exactly as confidence.py computed it.
#
# Fully deterministic: same inputs always produce the same output, no
# randomness, no ML, no new API field (the two keys it feeds already
# exist in every return block).
# SPRINT E ADDITION (Confidence Presentation Calibration, Goal 3):
# see `_academic_sections_complete()`'s own docstring for full
# rationale. Placed here, immediately before `_refine_response_
# confidence`, since it exists solely to feed that function's new
# `academic_sections_complete` parameter.
def _academic_sections_complete(mentor_response: str, is_comparison: bool) -> bool:
    """
    Deterministic, regex/keyword-only check for whether `mentor_response`
    actually contains every labeled section the Sprint E academic
    instructions (`_academic_comparison_body` / `_academic_normal_body`
    in `_build_output_format_instructions`) required for this answer
    shape. No ML, no embeddings, no model call -- plain text in, plain
    bool out, same style as `prompt_builder.detect_comparison_question`.

    This does NOT check whether the *content* under each heading is
    correct or well-grounded (that's the LLM's job, guided by the
    hallucination-safety instructions already in place) -- it only
    checks that the required structural headings are present, which is
    exactly the "no section omitted" half of the Goal 2 checklist, done
    in code as a trustworthy backstop rather than trusting the model's
    own silent self-check.

    Required headings:
      - comparison answers: Definition, Key Differences, Applications,
        Example.
      - normal answers: Definition, Explanation, Example.

    A heading counts as present only if the heading text itself
    appears. A response that instead used the "the retrieved material
    does not cover this aspect." disclaimer for a section is an honest,
    safe outcome -- but deliberately still returns False here, since
    the confidence boost this feeds (see `_refine_response_confidence`)
    is reserved for genuinely complete answers, not ones that had to
    decline a section for lack of evidence.
    """
    if not mentor_response:
        return False

    text = mentor_response.lower()

    required_headings = (
        ["definition", r"key\s+differences", "applications", "example"]
        if is_comparison
        else ["definition", "explanation", "example"]
    )

    return all(re.search(heading, text) for heading in required_headings)


def _refine_response_confidence(
    confidence_result,
    *,
    max_similarity: float,
    retrieved_chunks: List[Dict[str, Any]],
    sources_used: List[str],
    should_retrieve: bool,
    retrieval_evidence_too_weak: bool,
    parse_method: Optional[str] = None,
    detected_intent: str = "general",
    academic_sections_complete: bool = False,
) -> Dict[str, Any]:
    """
    Blends confidence.py's own `normalized_score` with retrieval- and
    parsing-quality signals already computed earlier in `ask_mentor()`,
    to produce a display score that actually moves with how well THIS
    turn went, instead of sitting near a flat ~0.4 regardless of
    outcome.

    Each extra axis is weighted and only included when it applies:
      - "retrieval"/"sources" axes are skipped entirely when
        `should_retrieve` is False (emotional/motivation/celebration/
        social turns never attempt retrieval by design, so they
        shouldn't be penalized for having no chunks to show).
      - the "parse" axis is skipped when `parse_method` is None (the
        similarity-gated path never calls the LLM, so there is no
        parse outcome to score).
    Weights always renormalize to sum to 1, so the score stays in
    [0, 1] regardless of how many axes apply this turn.

    SPRINT E ADDITION (Goal 3, presentation refinement only): two new,
    defaulted keyword-only parameters, `detected_intent` ("general",
    matching every other function in this file's default) and
    `academic_sections_complete` (False). Any existing caller that
    omits them keeps getting exactly the axis-blended score computed
    above, unchanged. When BOTH are meaningfully set by the caller --
    i.e. this is an academic answer, it contains every required
    section (`_academic_sections_complete()`), real sources were cited
    (`sources_used` non-empty), and the LLM matched the structured
    marker contract without falling back (`parse_method` in
    {"strict", "lenient"}) -- the already-computed axis-blended score
    is raised to a high floor instead of left wherever the coarser
    axis blend happened to land. This is presentation-layer only: it
    does not call confidence.py again, does not change `confidence_
    result` or the evidence-category scoring behind it, and never
    lowers a score, only raises one that already reflects a
    genuinely complete, grounded, well-parsed academic answer.
    """
    axes: List[tuple] = [("base", confidence_result.normalized_score, 0.5)]

    if should_retrieve:
        span = max(1e-6, 1.0 - SIMILARITY_THRESHOLD)
        retrieval_quality = 0.0 if (retrieval_evidence_too_weak or not retrieved_chunks) else min(
            1.0, max(0.0, (max_similarity - SIMILARITY_THRESHOLD) / span)
        )
        sources_signal = 1.0 if sources_used else (0.5 if retrieved_chunks else 0.0)
        axes.append(("retrieval", retrieval_quality, 0.3))
        axes.append(("sources", sources_signal, 0.1))

    if parse_method is not None:
        parse_signal = {"strict": 1.0, "lenient": 0.8, "fallback": 0.5}.get(parse_method, 0.5)
        axes.append(("parse", parse_signal, 0.1))

    weight_sum = sum(weight for _, _, weight in axes)
    display_score = sum(value * weight for _, value, weight in axes) / weight_sum
    display_score = round(min(1.0, max(0.0, display_score)), 3)

    # SPRINT E ADDITION (Goal 3): presentation-only, upward-only
    # calibration for the specific case the sprint asked for -- a
    # complete, source-grounded, cleanly-parsed academic answer.
    # `academic_sections_complete` is computed by the caller via
    # `_academic_sections_complete()` (structural headings only,
    # never content correctness); `sources_used` and `parse_method`
    # are the SAME signals already fed into the axes above, just
    # re-checked here as a hard, all-or-nothing floor rather than a
    # weighted blend. Never lowers `display_score` -- only raises it.
    _academic_answer_qualifies_for_boost = (
        detected_intent == "academic"
        and academic_sections_complete
        and bool(sources_used)
        and parse_method in ("strict", "lenient")
    )
    if _academic_answer_qualifies_for_boost:
        display_score = round(max(display_score, 0.97), 3)

    # Only flag for review when the RECALIBRATED score itself is weak,
    # or when this turn genuinely had evidence too thin to answer on
    # (the same condition the similarity gate already uses) — instead
    # of inheriting confidence.py's own (currently over-triggering)
    # `needs_human_review`, which is exactly the field this sprint was
    # asked to make more meaningful.
    display_needs_human_review = (
        display_score < CONFIDENCE_THRESHOLD
        or (should_retrieve and retrieval_evidence_too_weak)
    )

    return {
        "score": display_score,
        "needs_human_review": display_needs_human_review,
    }


# =========================================================
# TIMING SUMMARY PRINTER
# =========================================================
# Print-only helper. Formats the collected stage durations into the
# fixed-width report requested for ops/perf debugging. Does not affect
# control flow, does not raise, does not touch the function's return
# value.

def _print_timing_summary(timings: Dict[str, float]) -> None:
    def _line(label: str, key: str) -> str:
        value = timings.get(key)
        value_str = f"{value:.2f} s" if value is not None else "n/a"
        return f"{label} {'.' * (24 - len(label))} {value_str}"

    print("========== AI Mentor Timing ==========")
    print(_line("Context building", "context_building"))
    print(_line("Risk", "risk"))
    print(_line("Persona", "persona"))
    print(_line("SHAP", "shap"))
    print(_line("Cognitive", "cognitive"))
    print(_line("History", "history"))
    print(_line("Intent classification", "intent_classification"))
    print(_line("Retrieval", "retrieval"))
    print(_line("Prompt build", "prompt_build"))
    print(_line("Confidence", "confidence"))
    print(_line("Retrieval safety", "retrieval_safety"))
    print(_line("LLM", "llm"))
    print(_line("Citations", "citations"))
    print(_line("Save history", "save_history"))
    print(_line("Reasoning summary", "reasoning_summary"))
    print(_line("TOTAL", "total"))
    print("=====================================")


# =========================================================
# SPRINT 1.6 ADDITION (SOCIAL CONVERSATION QUALITY)
# =========================================================
# `intent_classifier.py` (Sprint 1.1) already detects bare "greeting" /
# "gratitude" / "farewell" messages ("Hi", "Thank you.", "Bye.") as
# their own intents, and those three intents were ALREADY absent from
# both `_RETRIEVAL_INTENTS` and `_STRICT_GATE_INTENTS` below (same
# allowlist mechanism Sprint 1.4 used for "celebration") -- so
# retrieval and the similarity gate were already off for them before
# this sprint. What was NOT natural: a bare "Hi" still went through the
# FULL academic-mentor pipeline -- the entire risk/persona/SHAP/
# cognitive-state system prompt from `prompt_builder.build_system_
# prompt()`, plus the 3-marker structured-output contract asking the
# LLM for a conversational reply AND a student recommendation AND an
# educator recommendation. A small model asked to produce two grounded
# "recommendations" in response to "Hi" cannot help but sound stilted.
#
# `_SOCIAL_INTENTS` names that same three-intent set explicitly (making
# the routing intent-driven rather than an implicit byproduct of what's
# missing from the other two sets), and `_handle_social_intent()` gives
# these three intents their own short-circuit path: a small, dedicated
# system prompt asking ONLY for a brief, warm, natural reply (no
# markers, no recommendations, no evidence framing), called with the
# same LLM. This is a NEW BRANCH, not a change to the existing
# academic/career/general/emotional/motivation/celebration path --
# `_RETRIEVAL_INTENTS`, `_STRICT_GATE_INTENTS`, `build_system_prompt()`,
# `_build_output_format_instructions()`, `_parse_structured_response()`,
# and every other existing function/branch are untouched and still run
# exactly as before for every other intent.
#
# Compatibility:
#   - `ask_mentor`'s signature is unchanged.
#   - `intent_classifier.py` is not touched -- greeting/gratitude/
#     farewell detection already existed; only how mentor_service.py
#     ROUTES those intents changes.
#   - No API response key is added, removed, or renamed. This path
#     returns the exact same key set as every other success path.
#     `student_recommendation` / `educator_recommendation` are `None`
#     (present, just not applicable to a bare "Hi"/"Thanks"/"Bye" --
#     same pattern the error path already uses for these fields).
#     `confidence_score` / `max_similarity` are `0.0` and
#     `retrieval_safety_label` is `"SAFE"` since no retrieval evidence
#     was ever in play for this turn (nothing to be unsafe about).
#   - `mentor_history` is still written with the same table/columns.
#   - Retrieval and the similarity gate are still never invoked for
#     these intents (same as before this sprint) -- this change only
#     touches the PROMPT/branching, not the retrieval-gating logic
#     itself.
_SOCIAL_INTENTS = {"greeting", "gratitude", "farewell"}

_SOCIAL_INTENT_GUIDANCE = {
    "greeting": (
        "The student has just greeted you (e.g. \"Hi\", \"Hello\"). "
        "Reply with a brief, warm welcome and an open invitation for "
        "them to share what's on their mind. Do not ask about academic "
        "performance, risk, or coursework unless they bring it up."
    ),
    "gratitude": (
        "The student is thanking you (e.g. \"Thank you\", \"Thanks\"). "
        "Reply with a brief, warm acknowledgement that you're glad you "
        "could help. Do not restate advice or bring up new topics."
    ),
    "farewell": (
        "The student is ending the conversation (e.g. \"Bye\", "
        "\"See you\"). Reply with a brief, warm send-off and a "
        "genuine well-wish for their studies."
    ),
}


def _build_social_system_prompt(detected_intent: str) -> str:
    """
    A deliberately small, standalone system prompt for bare social
    turns -- NOT built from `prompt_builder.build_system_prompt()`.
    Skips risk/persona/SHAP/cognitive-state framing and the 3-marker
    structured-output contract entirely, since none of that helps a
    model produce a natural one- or two-sentence reply to "Hi" and
    asking for it anyway is exactly what made these replies feel
    unnatural before this sprint. `prompt_builder.py` itself is not
    modified or imported differently -- this is a separate, additive
    prompt used only on this new branch.
    """
    guidance = _SOCIAL_INTENT_GUIDANCE[detected_intent]
    return (
        "You are Aura, a warm and supportive AI mentor for students. "
        f"{guidance} "
        "Keep your reply to one or two short sentences, natural and "
        "conversational -- no bullet points, no markers, no "
        "recommendations, no follow-up questions about grades or risk."
    )


def _handle_social_intent(
    user_id: str,
    question: str,
    detected_intent: str,
    context: "StudentContext",
    timings: Dict[str, float],
    _total_start: float,
) -> Dict[str, Any]:
    """
    Short-circuit path for `detected_intent in _SOCIAL_INTENTS`. See
    the SPRINT 1.6 module note above `_SOCIAL_INTENTS` for full
    rationale. Retrieval and the similarity gate are simply never
    reached from this branch -- `ask_mentor` calls this function
    before either is invoked.
    """
    _t0 = time.perf_counter()
    system_prompt = _build_social_system_prompt(detected_intent)
    response = ollama.chat(
        model="qwen2.5:3b",
        options={"temperature": 0.6, "top_p": 0.9, "num_predict": 80},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    timings["llm"] = time.perf_counter() - _t0
    mentor_response = response["message"]["content"].strip()

    _t0 = time.perf_counter()
    supabase.table("mentor_history").insert({
        "user_id": user_id,
        "question": question,
        "response": mentor_response,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()
    timings["save_history"] = time.perf_counter() - _t0

    timings["total"] = time.perf_counter() - _total_start
    _print_timing_summary(timings)

    return {
        "status": "success",
        "question": question,
        "mentor_response": mentor_response,
        "persona": context.persona,
        "risk_score": context.risk_score,
        "risk_level": context.risk_level,

        # Not applicable to a bare social turn -- present but None,
        # same convention already used on the error path below.
        "student_recommendation": None,
        "educator_recommendation": None,
        "confidence": None,
        "confidence_rationale": None,
        "evidence_used": [],
        "shap_available": context.shap_available,
        "cognitive_state_available": context.cognitive_state_available,

        "evidence_profile": None,
        "conversation_plan": None,

        # No retrieval was ever attempted for this turn, so there is
        # nothing to flag as unsafe/uncertain -- "SAFE" and 0.0 reflect
        # that honestly rather than reusing academic-path defaults.
        "confidence_score": 0.0,
        "needs_human_review": False,
        "retrieval_safety_label": "SAFE",
        "sources_used": [],
        "max_similarity": 0.0,

        "student_risk_confidence": None,
        "mentor_response_confidence": None,

        "detected_intent": detected_intent,
    }


# =========================================================
# MAIN ENTRY POINT (public API — unchanged signature)
# =========================================================

def ask_mentor(user_id: str, question: str) -> Dict[str, Any]:
    """
    Main entry point used by the existing FastAPI route(s). Signature
    and core return keys are unchanged from V1 for backward
    compatibility; new fields are additive.

    TIMING: wrapped with time.perf_counter() around each major stage.
    Durations are collected into a local `timings` dict and printed via
    `_print_timing_summary` before returning, on both the success and
    error paths. No stage's inputs, outputs, or order of execution have
    been changed — only measurement was added around them.
    """

    timings: Dict[str, float] = {}
    _total_start = time.perf_counter()

    try:
        # -------------------------
        # 1. Gather context (profile, risk, persona, SHAP, cognitive
        #    state, history) — never fabricates missing optional data.
        # -------------------------
        _t0 = time.perf_counter()
        context: StudentContext = build_student_context(user_id)
        timings["context_building"] = time.perf_counter() - _t0

        # Pull the per-stage breakdown collected inside
        # build_student_context() (see context_builder.py). This is a
        # read-only lookup into an internal timing dict — it does not
        # affect context's data fields or any downstream logic.
        _context_timings = getattr(context, "_timings", {}) or {}
        timings["risk"] = _context_timings.get("risk", 0.0)
        timings["persona"] = _context_timings.get("persona", 0.0)
        timings["shap"] = _context_timings.get("shap", 0.0)
        timings["cognitive"] = _context_timings.get("cognitive", 0.0)
        timings["history"] = _context_timings.get("history", 0.0)

        # -------------------------
        # 1b. MESSAGE ANALYSIS (Sprint 10 intent, Phase 2.9 emotion,
        #     Phase 3.1 emotion intensity, Phase 3.2 emotional
        #     trajectory) -- routes BEFORE retrieval.
        # -------------------------
        # PHASE 2.9 UPDATE: `classify_intent(question)` alone is
        # replaced by `analyze_message(question)`, the unified
        # deterministic Message Analysis Module in intent_classifier.py.
        # It parses the SAME question text once and returns both
        # `intent` (unchanged behaviour/values -- still drives the
        # retrieval routing below exactly as before) and the `emotion`
        # value. `detected_emotion` is used ONLY for prompt framing
        # (step 3) -- it does not affect retrieval routing, the
        # similarity gate, confidence, or recommendation selection, and
        # it is never stored or returned in the API response (see the
        # return dicts below, unchanged from Sprint 10).
        #
        # PHASE 3.1 UPDATE: `analyze_message()`'s return dict gains one
        # more key, `emotion_intensity` (Low/Medium/High), computed
        # deterministically from the same question text plus the
        # already-detected `emotion` -- see
        # intent_classifier.classify_emotion_intensity(). Exactly like
        # `detected_emotion`, `detected_emotion_intensity` is used ONLY
        # for prompt framing (step 3, pacing/validation guidance) --
        # it has no path into retrieval, the similarity gate,
        # confidence, evidence prioritization, SHAP reasoning, persona,
        # or recommendation selection, and it is never stored or
        # returned in the API response.
        #
        # PHASE 3.2 UPDATE: `analyze_message()` now also accepts the
        # OPTIONAL `chat_history` argument -- `context.chat_history`,
        # already built in step 1 above, read-only, no new query or
        # storage -- and returns one more key, `emotion_trajectory`
        # (Stable Emotional Distress / Escalating Distress / Improving
        # Emotional State / Positive Stability / Neutral / Mixed
        # Emotional Pattern), computed deterministically by
        # intent_classifier.classify_emotional_trajectory() from the
        # recent turns' emotion labels plus this turn's `emotion`.
        # Exactly like `detected_emotion` / `detected_emotion_intensity`,
        # `detected_emotion_trajectory` is used ONLY for prompt framing
        # (step 3, continuity-aware emotional framing) -- it has no
        # path into retrieval, the similarity gate, confidence,
        # evidence prioritization, SHAP reasoning, persona, or
        # recommendation selection, and it is never stored or returned
        # in the API response.
        _t0 = time.perf_counter()
        _message_analysis = analyze_message(
            question, getattr(context, "chat_history", "") or ""
        )
        detected_intent = _message_analysis["intent"]
        detected_emotion = _message_analysis["emotion"]
        detected_emotion_intensity = _message_analysis["emotion_intensity"]
        detected_emotion_trajectory = _message_analysis["emotion_trajectory"]
        # SPRINT 1.3 ADDITION (mixed-intent support): an independent
        # boolean read of the SAME question text -- see
        # intent_classifier.has_academic_reference() -- answering
        # "does this message ALSO reference academic content, on top
        # of whatever `detected_intent` resolved to." Used ONLY to
        # widen the retrieval decision below for "emotional"-labelled
        # messages that are also academic-shaped (e.g. "I'm stressed
        # because I don't understand DBMS"); it never changes
        # `detected_intent` itself, so persona/prompt framing/
        # conversation goal all stay exactly as emotion-first as
        # before this sprint.
        secondary_academic_signal = _message_analysis.get(
            "secondary_academic_signal", False
        )
        # SPRINT 1.5 ADDITION (Positive Learning Intent): a fourth
        # independent read of the SAME question text -- see
        # intent_classifier.extract_academic_topic() -- naming the
        # specific academic concept a celebration message references
        # (e.g. "recursion", "DBMS"), if any was deterministically
        # found. Like `secondary_academic_signal`, this never changes
        # `detected_intent` and is never used to relabel or override
        # it. It is read below ONLY when `detected_intent ==
        # "celebration"`, purely to let the system prompt name the
        # concept being celebrated -- it has no path into retrieval,
        # the similarity gate, confidence, or the API response schema.
        celebrated_academic_topic = _message_analysis.get("academic_topic")
        # SPRINT B ADDITION (Recommendation Relevance): the SAME value
        # above, reused under a clearer name for its second use site
        # (recommendation contextualization) so it isn't misread as
        # celebration-only. No new extraction call, no new signal --
        # this is exactly `_message_analysis.get("academic_topic")`,
        # already computed once per request regardless of intent.
        academic_topic_for_recommendations = celebrated_academic_topic
        # SPRINT D ADDITION (Academic Answer Excellence): a fifth
        # independent, deterministic read of the SAME question text --
        # see `prompt_builder.detect_comparison_question()` -- answering
        # "is this a comparison-style question (e.g. 'difference between
        # BFS and DFS', 'stack vs queue')?" Like `secondary_academic_
        # signal` / `celebrated_academic_topic` above, this NEVER changes
        # `detected_intent` and has no path into retrieval, the
        # similarity gate, confidence, persona, or the API response
        # schema. It is read below ONLY when `detected_intent ==
        # "academic"`, purely to pick which of two deterministic
        # academic-answer STRUCTURES `_build_output_format_instructions`
        # uses (comparison vs. normal).
        is_comparison_question = detect_comparison_question(question)
        timings["intent_classification"] = time.perf_counter() - _t0

        # SPRINT 2 ADDITION (Short-Term Context Relevance Gating): a
        # deterministic read of the SAME `detected_intent` just
        # classified above -- see `_short_term_context_is_relevant()`.
        # Used below to (a) build a filtered copy of `context` for the
        # system prompt, and (b) suppress the memory-clause callback in
        # the recommendation rationale. It does not change
        # `detected_intent` itself and has no path into retrieval, the
        # similarity gate, confidence, or persona selection.
        short_term_context_relevant = _short_term_context_is_relevant(detected_intent)

        # SPRINT 1.6 ADDITION (Social Conversation Quality): bare
        # greeting/gratitude/farewell turns take their own dedicated,
        # lightweight path -- see `_SOCIAL_INTENTS` / `_handle_social_
        # intent()` above for full rationale. This runs BEFORE the
        # retrieval-routing block below, so retrieval and the
        # similarity gate are never reached for these intents (in
        # addition to already being excluded from `_RETRIEVAL_INTENTS`
        # / `_STRICT_GATE_INTENTS`). Every other intent falls through
        # to the existing pipeline, completely unchanged.
        if detected_intent in _SOCIAL_INTENTS:
            return _handle_social_intent(
                user_id, question, detected_intent, context, timings, _total_start
            )

        # Which intents attempt retrieval at all. "emotional" and
        # "motivation" are deliberately excluded by default:
        # retrieving-then-discarding course chunks is not good enough,
        # because it still pays the retrieval cost and still risks a
        # stray chunk leaking into confidence/reasoning-layer signals
        # downstream. Skipping retrieval outright is the only way to
        # guarantee academic material (DBMS, OS, etc.) can never
        # surface in a purely emotional or motivation conversation.
        #
        # SPRINT 1.4 ADDITION (Celebration Routing): "celebration" is
        # likewise deliberately NOT added here. A message like "I
        # passed my exam!" or "I got placed!" has nothing to retrieve
        # against -- there is no course-material question to ground --
        # so attempting retrieval for it only risks either a weak/no
        # match (which, for a strict-gate intent, would incorrectly
        # produce INSUFFICIENT_EVIDENCE_MESSAGE for what is really a
        # congratulations) or a spurious topical match that pulls the
        # LLM toward tutoring content instead of acknowledging the
        # win. `detected_intent == "celebration"` therefore takes
        # exactly the same no-retrieval path as "emotional" /
        # "motivation" below, purely by NOT being listed in either
        # set -- no new branch, no new function, no change to
        # `should_retrieve` / `enforce_strict_gate`'s logic itself.
        _RETRIEVAL_INTENTS = {"academic", "career", "general"}
        # Which intents keep the ORIGINAL hard similarity-threshold gate
        # (i.e. can refuse to call the LLM at all on weak evidence).
        # "career" retrieval is best-effort: a weak/no match should
        # never stop the mentor from answering a career question
        # directly, it should just mean no sources get cited.
        #
        # SPRINT 1.4: "celebration" is excluded here for the same
        # reason it is excluded from `_RETRIEVAL_INTENTS` above -- with
        # `should_retrieve` already False for celebration,
        # `retrieved_chunks` is always `[]` and `enforce_strict_gate`
        # being False means the similarity-gate return block (which is
        # ONLY reachable via `if enforce_strict_gate and
        # retrieval_evidence_too_weak:`) is never entered, so
        # celebration messages always fall through to the normal
        # system-prompt -> LLM -> structured-parsing path below,
        # producing a real generated response instead of
        # INSUFFICIENT_EVIDENCE_MESSAGE.
        _STRICT_GATE_INTENTS = {"academic", "general"}

        # SPRINT 1.3 ADDITION (mixed-intent support): if this turn's
        # PRIMARY label is "emotional" but the message ALSO carries an
        # academic-content reference (secondary_academic_signal),
        # retrieval is allowed to run -- best-effort, exactly like
        # "career" below -- instead of being skipped outright. This is
        # the ONLY behavioural change in this sprint: a pure emotional
        # message with no academic reference ("I feel hopeless, I want
        # to give up") is completely unaffected and still skips
        # retrieval exactly as before.
        should_retrieve = detected_intent in _RETRIEVAL_INTENTS or (
            detected_intent == "emotional" and secondary_academic_signal
        )
        # Deliberately NOT extended for the mixed-intent case: an
        # emotional conversation must never be blocked or have its LLM
        # call refused because retrieved evidence was weak (that
        # would defeat the point of acknowledging the emotion at all).
        # So `enforce_strict_gate` stays keyed on `detected_intent`
        # alone -- "emotional" is never in `_STRICT_GATE_INTENTS`,
        # mixed or not.
        enforce_strict_gate = detected_intent in _STRICT_GATE_INTENTS

        # -------------------------
        # 2. Retrieve relevant learning material (only on intents whose
        #    execution path calls for it — see routing above)
        # -------------------------
        if should_retrieve:
            _t0 = time.perf_counter()
            retriever = get_retriever()
            timings["get_retriever"] = time.perf_counter() - _t0

            _t0 = time.perf_counter()
            retrieved_chunks = retriever.retrieve(question, k=TOP_K)
            timings["retrieval"] = time.perf_counter() - _t0
        else:
            # Emotional / motivation paths: retrieval never runs, so
            # there is no chunk list for academic material to hide in.
            retrieved_chunks = []
            timings["get_retriever"] = 0.0
            timings["retrieval"] = 0.0

        context.retrieved_chunks = retrieved_chunks

        # -------------------------
        # 2b. SIMILARITY THRESHOLD GATE (Hallucination Safety, Sprint 8;
        #     scope narrowed to strict-gate intents in Sprint 10)
        # -------------------------
        # `similarity_score` is computed once, in retrieval.py, from
        # the FAISS distance for each chunk (see that file's Sprint 8
        # update). We only read it here — nothing is recomputed. If
        # even the single strongest retrieved chunk falls below
        # SIMILARITY_THRESHOLD, the evidence is too weak to safely
        # ground any answer, so — for strict-gate intents — we refuse
        # to call the LLM at all rather than risk it improvising past
        # thin evidence (Req: Similarity Threshold).
        #
        # ISSUE 2 HARDENING (IEEE paper note): this `if` below is the
        # SOLE, AUTHORITATIVE gate on whether ollama.chat() is ever
        # invoked for this request. There is no second, independent
        # copy of this comparison anywhere else in the pipeline for
        # this request to fall through: when the condition is True we
        # `return` immediately, before system-prompt construction
        # (step 3) and before the LLM call (step 5) are ever reached.
        # `retrieval_safety_label` is still computed on this branch
        # (via classify_retrieval_safety, same call used on the
        # non-gated path below) purely for observability/labeling —
        # it does not additionally gate anything here, since weak
        # similarity alone is already sufficient grounds to refuse the
        # LLM call (Req: Similarity Threshold takes precedence over
        # the softer SAFE/UNCERTAIN/NEEDS_REVIEW label).
        similarity_scores = [
            chunk.get("similarity_score")
            for chunk in retrieved_chunks
            if isinstance(chunk.get("similarity_score"), (int, float))
        ]
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        retrieval_evidence_too_weak = max_similarity < SIMILARITY_THRESHOLD

        # SPRINT 10 ADDITION: for non-strict-gate intents (currently
        # only "career"), weak/absent evidence does NOT block the LLM
        # call — it just means this turn proceeds with no chunks to
        # cite. `select_chunks_for_prompt()` / the Sources block further
        # down will then correctly render "no sources", and
        # `_section_learning_material` already renders cleanly on an
        # empty list ("none retrieved for this query"), so no change
        # was needed in prompt_builder.py for this to work.
        if retrieval_evidence_too_weak and not enforce_strict_gate:
            retrieved_chunks = []
            context.retrieved_chunks = retrieved_chunks

        if enforce_strict_gate and retrieval_evidence_too_weak:
            _t0 = time.perf_counter()
            confidence_result = assess_confidence(
                context,
                retrieved_chunks_found=len(retrieved_chunks) > 0,
                retrieved_chunks=retrieved_chunks,
            )
            timings["confidence"] = time.perf_counter() - _t0

            focus = determine_priority_focus(context)
            evidence_profile = build_evidence_profile(context, focus)
            conversation_plan = build_conversation_plan(
                context,
                focus,
                get_persona_actions(context.persona),
                get_psychological_framework(context.persona),
            )
            retrieval_safety_label = classify_retrieval_safety(
                max_similarity=max_similarity,
                normalized_confidence=confidence_result.normalized_score,
                contradiction_detected=confidence_result.contradiction_detected,
                similarity_threshold=SIMILARITY_THRESHOLD,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )

            mentor_response = INSUFFICIENT_EVIDENCE_MESSAGE

            # SPRINT 11 ADDITION: recalibrated display score for this
            # gated turn. No LLM ran, so `parse_method` is omitted
            # (that axis is skipped); retrieval/sources axes correctly
            # collapse to their weakest values here since this branch
            # only exists because retrieval evidence was too thin.
            _refined_confidence = _refine_response_confidence(
                confidence_result,
                max_similarity=max_similarity,
                retrieved_chunks=retrieved_chunks,
                sources_used=[],
                should_retrieve=should_retrieve,
                retrieval_evidence_too_weak=retrieval_evidence_too_weak,
            )

            # Persist history the same way as a normal turn — same
            # table, same columns, no schema change — so this gated
            # response is still part of the conversation record.
            supabase.table("mentor_history").insert({
                "user_id": user_id,
                "question": question,
                "response": mentor_response,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()

            timings["total"] = time.perf_counter() - _total_start
            _print_timing_summary(timings)

            return {
                "status": "success",
                "question": question,
                "mentor_response": mentor_response,
                "persona": context.persona,
                "risk_score": context.risk_score,
                "risk_level": context.risk_level,

                "student_recommendation": (
                    "No confident, evidence-grounded recommendation could be "
                    "made for this specific question — try rephrasing it or "
                    "asking about a topic covered in the course material."
                ),
                "educator_recommendation": (
                    "Retrieved evidence for this question was too weak to "
                    "ground a response; educator review is advised before "
                    "acting on it."
                ),
                "confidence": confidence_result.level,
                "confidence_rationale": confidence_result.rationale,
                "evidence_used": confidence_result.evidence_used,
                "shap_available": context.shap_available,
                "cognitive_state_available": context.cognitive_state_available,

                "evidence_profile": evidence_profile,
                "conversation_plan": conversation_plan,

                # Hallucination Safety Hardening (Sprint 8) — additive.
                # SPRINT 11: value now comes from `_refine_response_
                # confidence()` above instead of the raw
                # `confidence_result.normalized_score` — see that
                # function's docstring. `needs_human_review` stays
                # hardcoded True: this branch's entire reason for
                # existing is evidence too weak to answer on, so it is
                # never in question here regardless of recalibration.
                "confidence_score": _refined_confidence["score"],
                "needs_human_review": True,
                "retrieval_safety_label": retrieval_safety_label,
                "sources_used": [],
                "max_similarity": max_similarity,

                # Issue 1 fix — unambiguous aliases (see success-path
                # return below for full explanation). On this gated
                # path mentor_response_confidence will typically be low
                # by construction, since it's exactly why the LLM was
                # never called.
                "student_risk_confidence": confidence_result.student_risk_confidence,
                "mentor_response_confidence": _refined_confidence["score"],

                # New in Sprint 10 (Intent-Aware Retrieval Routing) —
                # additive: which execution path this question took.
                "detected_intent": detected_intent,
            }

        # -------------------------
        # 2c. PHASE 3.3 ADDITION (Adaptive Conversation Planning):
        #     decide the CONVERSATIONAL GOAL for this turn, before the
        #     response is generated. `determine_conversation_goal()` is
        #     a deterministic, side-effect-free function of persona,
        #     the priority focus already computed for this turn's
        #     evidence, and the intent/emotion/intensity/trajectory
        #     values already computed in step 1b -- no new evidence is
        #     gathered here. `conversation_goal` is used ONLY to (a)
        #     render one short new prompt section (see
        #     prompt_builder._section_conversation_goal) and (b),
        #     optionally, influence the ORDERING (never the selection
        #     or content) of existing conversational slots in
        #     recommendation_reasoning.py. It is never returned in the
        #     API response.
        # -------------------------
        _t0 = time.perf_counter()
        _focus_for_goal = determine_priority_focus(context)
        conversation_goal = determine_conversation_goal(
            context.persona,
            _focus_for_goal,
            detected_intent,
            detected_emotion,
            detected_emotion_intensity,
            detected_emotion_trajectory,
        )
        timings["conversation_goal"] = time.perf_counter() - _t0

        # -------------------------
        # 3. Build structured system prompt
        # -------------------------
        _t0 = time.perf_counter()

        # SPRINT 2 ADDITION (Short-Term Context Relevance Gating): a
        # shallow copy of `context` whose `chat_history` is cleared
        # when `short_term_context_relevant` is False, so
        # `build_system_prompt()` never sees short-term chat history
        # judged not relevant to this turn (e.g. an unrelated earlier
        # turn's topic bleeding into a fresh exam question). The
        # original `context` object is left completely untouched --
        # every other stage (retrieval, confidence, database write,
        # fallback recommendations) still operates on the real,
        # unfiltered context.
        context_for_prompt = copy.copy(context)
        if not short_term_context_relevant:
            context_for_prompt.chat_history = []

        system_prompt = build_system_prompt(
            context_for_prompt,
            retrieved_chunks,
            detected_intent,
            detected_emotion,
            detected_emotion_intensity,
            detected_emotion_trajectory,
            conversation_goal,
        )
        # SPRINT 1.5 ADDITION (Positive Learning Intent): a small,
        # deterministic, additive string appended to `build_system_
        # prompt()`'s own output -- the exact same technique already
        # used just below for `_build_output_format_instructions()` --
        # rather than a change to `build_system_prompt()` itself, so
        # prompt_builder.py stays completely untouched. Fires ONLY when
        # this turn is a celebration AND a specific academic concept
        # was deterministically found in the question (see
        # intent_classifier.extract_academic_topic()). It does not
        # introduce a new intent, does not turn retrieval or the
        # similarity gate back on for celebration (both stay OFF, per
        # `_RETRIEVAL_INTENTS` / `_STRICT_GATE_INTENTS` above, which
        # are untouched), and does not add any new key to the API
        # response -- it only shapes what the LLM is told to celebrate
        # by name.
        if detected_intent == "celebration" and celebrated_academic_topic:
            system_prompt += (
                "\n\nCELEBRATION CONTEXT: The student is celebrating "
                f'progress specifically on "{celebrated_academic_topic}". '
                "Acknowledge and celebrate this specific win by name in "
                "your reply. Do not quiz, re-teach, or retrieve "
                "additional material about it -- this remains a "
                "celebration, not a content lookup."
            )
        system_prompt += _build_output_format_instructions(
            context.persona, detected_intent, is_comparison_question
        )
        timings["prompt_build"] = time.perf_counter() - _t0

        # -------------------------
        # 4. Confidence — derived strictly from available evidence,
        #    computed independently of the LLM call.
        # -------------------------
        _t0 = time.perf_counter()
        confidence_result = assess_confidence(
            context,
            retrieved_chunks_found=len(retrieved_chunks) > 0,
            retrieved_chunks=retrieved_chunks,
        )
        timings["confidence"] = time.perf_counter() - _t0

        # -------------------------
        # 4b. Retrieval safety label (Hallucination Safety, Sprint 8)
        # -------------------------
        # Combines the similarity we already gated on above with the
        # normalized confidence and contradiction flag just computed,
        # into one of SAFE / UNCERTAIN / NEEDS_REVIEW (Req: Reasoning
        # Layer). Computed here (before the LLM call) since it depends
        # only on retrieval + confidence, not on the model's output.
        _t0 = time.perf_counter()
        retrieval_safety_label = classify_retrieval_safety(
            max_similarity=max_similarity,
            normalized_confidence=confidence_result.normalized_score,
            contradiction_detected=confidence_result.contradiction_detected,
            similarity_threshold=SIMILARITY_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        timings["retrieval_safety"] = time.perf_counter() - _t0

        # -------------------------
        # 5. Call the LLM
        # -------------------------
        _t0 = time.perf_counter()
        print("=" * 60)
        print("SYSTEM PROMPT LENGTH:", len(system_prompt))
        print("USER QUESTION LENGTH:", len(question))
        print("=" * 60)
        # SPRINT 5 (structured-output reliability):
        # - num_predict raised 250 -> 480. The failure analysis showed
        #   the model consistently failing to emit the last one or two
        #   markers; 250 tokens is tight for a warm reply PLUS two
        #   recommendation sections from a verbose 3B model, and a
        #   token budget that runs out mid-format is functionally
        #   indistinguishable from "ignored the format" once parsed.
        #   This is a runtime sampling parameter, not an API/schema
        #   change, and raises latency only modestly for a 3B model.
        # - temperature lowered 0.7 -> 0.5. Lower temperature improves
        #   adherence to an explicit structural instruction (like exact
        #   marker reproduction) without materially flattening the
        #   conversational warmth of the reply itself, which is a
        #   content property, not a sampling-randomness property.
        # SPRINT A ADDITION (Academic Answer Quality): a tutor-depth
        # academic answer (definition + explanation + differences/
        # steps/example) needs materially more output tokens than a
        # short reply, and was previously competing for the same fixed
        # 480-token budget as the two mandatory recommendation
        # sections -- a likely contributor to the reported "1-2
        # sentence answer" symptom. Only `detected_intent == "academic"`
        # gets a larger budget (480 -> 650); every other intent's
        # num_predict, temperature, and top_p are byte-for-byte
        # unchanged from before.
        _num_predict = 650 if detected_intent == "academic" else 480
        response = ollama.chat(
            model="qwen2.5:3b",
            options={"temperature": 0.5, "top_p": 0.9, "num_predict": _num_predict},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        timings["llm"] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        raw_text = response["message"]["content"].strip()
        # `context` is passed so that IF the model still misses the
        # structured markers, the two recommendation fields are built
        # from real evidence already gathered in this request (see
        # _fallback_recommendations) instead of a static placeholder.
        parsed = _parse_structured_response(
            raw_text, context, detected_intent, question, conversation_goal,
            short_term_context_relevant, academic_topic_for_recommendations,
        )
        timings["parse_response"] = time.perf_counter() - _t0
        # SPRINT 11 ADDITION: internal-only signal (see
        # `_parse_structured_response`'s `_parse_method` key and
        # `_refine_response_confidence()`'s docstring) — never itself
        # part of the API response.
        parse_method = parsed.get("_parse_method")

        # -------------------------
        # 5b. Evidence citations (Hallucination Safety, Sprint 8)
        # -------------------------
        # `select_chunks_for_prompt()` returns EXACTLY the chunks
        # prompt_builder.py rendered into the Learning Material
        # section this turn — reusing that function (rather than
        # re-slicing retrieved_chunks independently here) guarantees
        # the "Sources:" list below can never cite more, fewer, or
        # different sources than what actually grounded the answer
        # (Req: Evidence Citations). This is appended in CODE, not
        # parsed out of the LLM's own text, so a small model's
        # occasional tendency to misname or invent a source can never
        # reach the citation list shown to the user.
        _t0 = time.perf_counter()
        cited_chunks = select_chunks_for_prompt(retrieved_chunks)
        source_names = _dedupe_preserve_order(
            [chunk.get("source") or chunk.get("topic") for chunk in cited_chunks]
        )
        parsed["mentor_response"] = parsed["mentor_response"] + _build_sources_block(source_names)
        timings["citations"] = time.perf_counter() - _t0

        # SPRINT 11 ADDITION: recalibrated response-grounding confidence
        # for this successful turn, using the retrieval margin, whether
        # real sources were cited, and whether the LLM matched the
        # structured contract — see `_refine_response_confidence()`.
        # SPRINT E ADDITION (Goal 3): structural-only, deterministic
        # check of whether this academic answer actually contains
        # every required labeled section (see
        # `_academic_sections_complete()`). Computed for every intent
        # (cheap, pure string check) but only ever matters below when
        # `detected_intent == "academic"` -- `_refine_response_
        # confidence` itself also re-checks `detected_intent`, so this
        # is inert for every non-academic turn regardless.
        _academic_sections_ok = _academic_sections_complete(
            parsed["mentor_response"], is_comparison_question
        )

        _refined_confidence = _refine_response_confidence(
            confidence_result,
            max_similarity=max_similarity,
            retrieved_chunks=retrieved_chunks,
            sources_used=source_names,
            should_retrieve=should_retrieve,
            retrieval_evidence_too_weak=retrieval_evidence_too_weak,
            parse_method=parse_method,
            detected_intent=detected_intent,
            academic_sections_complete=_academic_sections_ok,
        )

        # -------------------------
        # 6. Persist chat history (same table/columns as V1 — no
        #    schema change; we only store the conversational reply,
        #    matching original behaviour).
        # -------------------------
        _t0 = time.perf_counter()
        supabase.table("mentor_history").insert({
            "user_id": user_id,
            "question": question,
            "response": parsed["mentor_response"],
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        timings["save_history"] = time.perf_counter() - _t0

        # -------------------------
        # 7b. SPRINT 7 (additive only): mirror the same PRIORITY FOCUS
        #     / Evidence Fusion / Conversation Plan the prompt already
        #     told the LLM to anchor on, so API consumers can see WHY
        #     without needing raw scores. Reuses the exact same
        #     `determine_priority_focus` this request's prompt and
        #     fallback recommendations already used — never a second,
        #     independent computation.
        # -------------------------
        _t0 = time.perf_counter()
        _focus_for_response = determine_priority_focus(context)
        evidence_profile = build_evidence_profile(context, _focus_for_response)
        conversation_plan = build_conversation_plan(
            context,
            _focus_for_response,
            get_persona_actions(context.persona),
            get_psychological_framework(context.persona),
        )
        timings["reasoning_summary"] = time.perf_counter() - _t0

        timings["total"] = time.perf_counter() - _total_start
        _print_timing_summary(timings)

        # -------------------------
        # 8. Return — old keys preserved, new keys additive.
        # -------------------------
        return {
            "status": "success",
            "question": question,
            "mentor_response": parsed["mentor_response"],
            "persona": context.persona,
            "risk_score": context.risk_score,
            "risk_level": context.risk_level,

            # New in V2:
            # NOTE (Issue 1 fix — confidence-field disambiguation):
            # "confidence" here is the STUDENT's psychological/risk-
            # assessment confidence (High/Medium/Low), NOT a measure of
            # how well-grounded this particular mentor response is.
            # Kept under its original name/value for backward
            # compatibility with existing callers; see
            # "student_risk_confidence" below for the same value under
            # an unambiguous name.
            "student_recommendation": parsed["student_recommendation"],
            "educator_recommendation": parsed["educator_recommendation"],
            "confidence": confidence_result.level,
            "confidence_rationale": confidence_result.rationale,
            "evidence_used": confidence_result.evidence_used,
            "shap_available": context.shap_available,
            "cognitive_state_available": context.cognitive_state_available,

            # New in Sprint 7 (purely additive — safe for callers to ignore):
            "evidence_profile": evidence_profile,
            "conversation_plan": conversation_plan,

            # New in Sprint 8 (Hallucination Safety Hardening) — additive:
            # NOTE (Issue 1 fix): "confidence_score" is the MENTOR
            # RESPONSE's retrieval-grounding confidence (0..1) — a
            # completely different axis from "confidence" above. Kept
            # under its original name/value for backward compatibility;
            # see "mentor_response_confidence" below for the same value
            # under an unambiguous name.
            # SPRINT 11: value now comes from `_refine_response_
            # confidence()` above instead of the raw
            # `confidence_result.normalized_score` / `.needs_human_
            # review` — see that function's docstring for why. The
            # `retrieval_safety_label == "NEEDS_REVIEW"` check is kept
            # alongside it (reasoning_layer.py is untouched, so its own
            # SAFE/UNCERTAIN/NEEDS_REVIEW verdict still always wins).
            "confidence_score": _refined_confidence["score"],
            "needs_human_review": (
                _refined_confidence["needs_human_review"]
                or retrieval_safety_label == "NEEDS_REVIEW"
            ),
            "retrieval_safety_label": retrieval_safety_label,
            "sources_used": source_names,
            "max_similarity": max_similarity,

            # -------------------------------------------------------
            # Issue 1 fix — unambiguous, non-colliding field names.
            # These are pure aliases of "confidence" and
            # "confidence_score" above (same values, same source of
            # truth: confidence_result). Added, not substituted, so
            # existing consumers of "confidence"/"confidence_score"
            # are unaffected. New/updated consumers should prefer
            # these two names going forward:
            #   student_risk_confidence   -> High/Medium/Low, about the
            #                                learner's risk assessment.
            #   mentor_response_confidence -> 0..1, about whether THIS
            #                                answer was well-grounded
            #                                in retrieved evidence.
            # -------------------------------------------------------
            "student_risk_confidence": confidence_result.student_risk_confidence,
            "mentor_response_confidence": _refined_confidence["score"],

            # New in Sprint 10 (Intent-Aware Retrieval Routing) —
            # additive: which execution path this question took
            # (academic/career keep retrieval, emotional/motivation
            # skip it entirely — see intent_classifier.py).
            "detected_intent": detected_intent,
        }

    except Exception as e:
        timings["total"] = time.perf_counter() - _total_start
        _print_timing_summary(timings)

        return {
            "status": "error",
            "mentor_response": (
                "I'm having trouble thinking right now. Please try "
                "again in a moment."
            ),
            "message": str(e),

            # Keep new keys present-but-null on error too, so callers
            # that started depending on them don't need extra None
            # checks for the error path.
            "student_recommendation": None,
            "educator_recommendation": None,
            "confidence": None,
            "confidence_rationale": None,
            "evidence_used": [],
            "shap_available": False,
            "cognitive_state_available": False,

            # Sprint 7 additive keys, present-but-null on error too.
            "evidence_profile": None,
            "conversation_plan": None,

            # Sprint 8 additive keys (Hallucination Safety Hardening).
            # needs_human_review defaults True and retrieval_safety_label
            # defaults "NEEDS_REVIEW" on the error path, since a failed
            # request has no verified evidence behind it at all — this
            # is a conservative default, not a computed assessment.
            "confidence_score": None,
            "needs_human_review": True,
            "retrieval_safety_label": "NEEDS_REVIEW",
            "sources_used": [],
            "max_similarity": None,

            # Issue 1 fix — unambiguous aliases, null on the error path
            # like every other confidence-related field here (no
            # verified evidence exists behind a failed request).
            "student_risk_confidence": None,
            "mentor_response_confidence": None,

            # Sprint 10 additive key. `detected_intent` may not exist
            # yet if the exception happened before classification ran
            # (e.g. during context-building) — `locals().get(...)`
            # returns None in that case rather than raising a second,
            # masking NameError on top of the original failure.
            "detected_intent": locals().get("detected_intent"),
        }
