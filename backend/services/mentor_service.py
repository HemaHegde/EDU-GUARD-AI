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

import re
import time
from datetime import datetime
from typing import Dict, Any, Optional

import ollama

from config.supabase_client import supabase

from .context_builder import build_student_context, StudentContext
from .prompt_builder import (
    build_system_prompt,
    determine_priority_focus,
    select_chunks_for_prompt,
)
from .retrieval import get_retriever, TOP_K, SIMILARITY_THRESHOLD
from .confidence import assess_confidence, CONFIDENCE_THRESHOLD
from .reasoning_layer import (
    get_few_shot_example,
    build_evidence_profile,
    build_conversation_plan,
    classify_retrieval_safety,
)
from .prompt_builder import get_persona_actions, get_psychological_framework

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


def _build_output_format_instructions(persona: str = "Unknown Persona") -> str:
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
_HUMAN_REASON_BY_FOCUS_ID = {
    "enrichment": "steady performance and no elevated risk",
    "confusion": "elevated confusion signals in recent activity",
    "burnout": "a severe burnout risk pattern",
    "engagement": "low recent engagement",
    "inactivity": "inactivity being the strongest measured risk driver",
    "persona_default": "the learner's overall behavioural pattern",
}


def _fallback_recommendations(context: Optional["StudentContext"]) -> Dict[str, str]:
    """
    SPRINT 5 ADDITION, updated in SPRINT 6 to use the same evidence-
    prioritization logic as the prompt itself.

    Builds a Student Recommendation and Educator Recommendation
    deterministically from data ALREADY in `context`, for use only
    when the LLM fails to produce the structured markers (both the
    strict and lenient parse attempts miss). This directly replaces
    the previous static "Not available" placeholder text, which is
    what caused every single scenario in the AI Mentor Failure
    Analysis to be logged as "Partial Success" even though the
    conversational reply itself was fine.

    SPRINT 6 CHANGE: previously this pulled the top SHAP feature and
    persona action independently, which could disagree with whatever
    the LLM was actually told to prioritize in the prompt (e.g. SHAP
    inactivity, when confusion or burnout was the real dominant
    signal). It now calls `determine_priority_focus()` — the exact
    same function that produces the prompt's PRIORITY FOCUS section —
    so a fallback recommendation reasons about evidence the same way
    the LLM was instructed to, keeping the two modes consistent with
    each other. This is still not the LLM inventing content, and still
    not a new computation: `determine_priority_focus` only reads
    fields already gathered by context_builder.py.
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
    human_reason = _HUMAN_REASON_BY_FOCUS_ID.get(focus["id"], "the available evidence")

    student_recommendation = (
        f"Based on {human_reason}, {focus['student_action']}."
    )
    educator_recommendation = (
        f"Based on {human_reason}, {focus['educator_action']}."
    )

    return {
        "student_recommendation": student_recommendation,
        "educator_recommendation": educator_recommendation,
    }


def _parse_structured_response(
    raw_text: str,
    context: Optional["StudentContext"] = None,
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
    """

    match = _STRICT_PATTERN.search(raw_text)
    if match:
        reply, student_rec, educator_rec = match.groups()
        return {
            "mentor_response": reply.strip(),
            "student_recommendation": student_rec.strip(),
            "educator_recommendation": educator_rec.strip(),
        }

    lenient_match = _LENIENT_PATTERN.search(raw_text)
    if lenient_match:
        reply, student_rec, educator_rec = lenient_match.groups()
        return {
            "mentor_response": reply.strip(),
            "student_recommendation": student_rec.strip() or _fallback_recommendations(context)["student_recommendation"],
            "educator_recommendation": educator_rec.strip() or _fallback_recommendations(context)["educator_recommendation"],
        }

    # Neither pattern matched at all. Don't crash, don't invent the
    # conversational reply — treat the whole raw text as the reply
    # (unchanged behaviour), but ground the two recommendations in
    # real evidence instead of a static placeholder.
    fallback = _fallback_recommendations(context)
    return {
        "mentor_response": raw_text.strip(),
        "student_recommendation": fallback["student_recommendation"],
        "educator_recommendation": fallback["educator_recommendation"],
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
        # 2. Retrieve relevant learning material
        # -------------------------
        _t0 = time.perf_counter()
        retriever = get_retriever()
        timings["get_retriever"] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        retrieved_chunks = retriever.retrieve(question, k=TOP_K)
        context.retrieved_chunks = retrieved_chunks
        timings["retrieval"] = time.perf_counter() - _t0

        # -------------------------
        # 2b. SIMILARITY THRESHOLD GATE (Hallucination Safety, Sprint 8)
        # -------------------------
        # `similarity_score` is computed once, in retrieval.py, from
        # the FAISS distance for each chunk (see that file's Sprint 8
        # update). We only read it here — nothing is recomputed. If
        # even the single strongest retrieved chunk falls below
        # SIMILARITY_THRESHOLD, the evidence is too weak to safely
        # ground any answer, so we refuse to call the LLM at all
        # rather than risk it improvising past thin evidence (Req:
        # Similarity Threshold).
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

        if retrieval_evidence_too_weak:
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

                # Hallucination Safety Hardening (Sprint 8) — additive:
                "confidence_score": confidence_result.normalized_score,
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
                "mentor_response_confidence": confidence_result.mentor_response_confidence,
            }

        # -------------------------
        # 3. Build structured system prompt
        # -------------------------
        _t0 = time.perf_counter()
        system_prompt = build_system_prompt(context, retrieved_chunks)
        system_prompt += _build_output_format_instructions(context.persona)
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
        response = ollama.chat(
            model="qwen2.5:3b",
            options={"temperature": 0.5, "top_p": 0.9, "num_predict": 480},
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
        parsed = _parse_structured_response(raw_text, context)
        timings["parse_response"] = time.perf_counter() - _t0

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
            "confidence_score": confidence_result.normalized_score,
            "needs_human_review": (
                confidence_result.needs_human_review
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
            "mentor_response_confidence": confidence_result.mentor_response_confidence,
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
        }
