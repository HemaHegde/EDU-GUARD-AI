#!/usr/bin/env python3
"""
ml/rag_ablation/rag_ablation.py

Real, end-to-end ablation evaluation of the AI Mentor pipeline
(mentor_service.ask_mentor + its real dependencies), for the EduGuard-AI
IEEE Access submission.

=============================================================================
WHAT THIS SCRIPT IS
=============================================================================
This replaces an earlier version of the RAG ablation study that evaluated
synthetic StudentContext fixtures. This version does not fabricate any
student context, persona, risk score, SHAP explanation, cognitive state,
retrieval result, mentor response, or confidence value. Every number in the
generated report is either:
    (a) read directly from a real backend call (build_student_context,
        ask_mentor, the retriever, assess_confidence, an actual Ollama
        response), or
    (b) explicitly marked "unavailable" / "not observed" when the
        corresponding real service could not be reached or did not return
        data.
Nothing is estimated, inferred, or invented to fill a gap.

=============================================================================
WHAT THIS SCRIPT DOES NOT DO
=============================================================================
- It does NOT modify mentor_service.py, prompt_builder.py, reasoning_layer.py,
  context_builder.py, confidence.py, retrieval.py, persona_service.py,
  risk_service.py, shap_service.py, cognitive_adapter.py, any API route, or
  the database schema. Those modules are only imported and called.
- It does NOT create synthetic StudentContext fixtures, hardcode risk
  scores/personas/SHAP/cognitive states, or hand-write recommendations.
- It does NOT pretend live LLM inference happened when it didn't.

=============================================================================
TWO EVALUATION MODES PER (user, question)
=============================================================================
1. "Full System" configuration
   -----------------------------
   Calls the real, unmodified `mentor_service.ask_mentor(user_id, question)`
   exactly as production code would. This is the only configuration that
   is a literal, byte-for-byte call into the real orchestrator. Because
   ask_mentor() is monolithic (it builds its own context internally and has
   no parameter for injecting a modified context), it cannot itself be used
   to run the masked-evidence ablation arms below.

   NOTE: ask_mentor() also writes a row to `mentor_history` on every call,
   exactly like production traffic. That side effect is real, not
   simulated by this script, and is disclosed here for reproducibility
   awareness -- running this evaluation against a production project will
   add rows to that table.

2. Ablation arms ("Without Retrieval", "Without Persona", "Without SHAP",
   "Without Cognitive", "Risk Only")
   -----------------------------------------------------------------------
   Because ask_mentor() cannot accept an injected context, these arms are
   produced by:
     (a) calling the REAL `build_student_context(user_id)` to get the
         student's real context (never synthetic),
     (b) deep-copying it and REMOVING (never replacing/substituting) the
         evidence fields that configuration is meant to ablate,
     (c) replaying the exact same sequence of REAL functions that
         ask_mentor() itself calls internally -- retrieval.get_retriever(),
         prompt_builder.build_system_prompt(), mentor_service's own
         (imported, not reimplemented) output-format instructions,
         confidence.assess_confidence(), and, if live inference is active,
         the real `ollama.chat(...)` call with mentor_service's actual
         model/parameters, followed by mentor_service's own
         `_parse_structured_response()`.
   No business logic (prompt content, confidence rubric, parsing, fallback
   recommendation logic) is reimplemented anywhere in this script -- every
   step calls the real function from the real module. This orchestration
   is a disclosed methodological necessity, not a hidden reimplementation;
   see the "Methodology" section of the generated report.

   To avoid polluting the real `mentor_history` table with ablation-arm
   traffic, ablation-arm calls do NOT write to mentor_history. Only the
   "Full System" arm (via the real, unmodified ask_mentor()) writes
   history, exactly as production does.

=============================================================================
LIVE VS. OFFLINE
=============================================================================
Controlled by USE_LIVE_LLM (env var, default True) and an actual reachability
probe against Ollama (`ollama.list()`), never assumed. If either is false,
every record produced in this run is explicitly marked
`live_llm_used: false`, and the "Full System" arm -- which has no
deterministic fallback inside ask_mentor() itself -- will honestly report
`status: "error"` for that arm rather than inventing an LLM-free result,
since ask_mentor() unconditionally calls the LLM. The ablation arms, by
contrast, CAN still report real retrieval/confidence/evidence metrics with
the LLM call skipped, and are labeled accordingly.

=============================================================================
REPRODUCIBILITY REQUIREMENTS
=============================================================================
This script must be run from within (or pointed at, via BACKEND_ROOT) the
actual EduGuard-AI backend project, so that `services.mentor_service` and
its siblings import successfully as a real Python package (i.e. `services/`
needs an `__init__.py`, or your project's actual package name -- see
`_add_backend_to_syspath()` below and adjust `_CANDIDATE_PACKAGE_NAMES` if
your project's import path differs from `services.*`).

Relevant environment variables:
    BACKEND_ROOT        Path to the folder containing the `services`
                         package + `config` package (supabase client).
                         If unset, the script searches upward from its own
                         location and falls back to backend_unavailable.
    EVAL_USER_IDS        Comma-separated real user IDs to evaluate.
                         If unset, the script tries an `evaluation_users`
                         table via Supabase; if that also yields nothing,
                         it exits without evaluating (never invents IDs).
    EVAL_QUESTIONS_FILE  Path to a newline-delimited file of real
                         evaluation questions. If unset, a small built-in
                         set of generic mentoring questions is used (these
                         are evaluation PROMPTS, not student data, so this
                         is not a fabrication of student context).
    USE_LIVE_LLM         "true"/"false" (default: true).
    OLLAMA_MODEL         Overrides the model name used for the ablation
                         arms' direct ollama.chat() call (default matches
                         mentor_service.py's "qwen2.5:3b" at the time this
                         script was written -- if the backend's model
                         changes, prefer letting the script read it from
                         mentor_service's source rather than editing this
                         default; see `_infer_model_name()`).

Usage:
    cd <project_root>            # wherever `services/` lives
    python ml/rag_ablation/rag_ablation.py \\
        --user-ids <id1>,<id2>,... \\
        --questions-file my_questions.txt

Outputs (all regenerated from this run's real data only):
    ml/rag_ablation/outputs/records.jsonl   (one real observation per line)
    ml/rag_ablation/outputs/summary.json    (aggregated, traceable metrics)
    ml/rag_ablation/figures/*.png           (only for metrics with real data)
    ml/rag_ablation/reports/rag_ablation_report.md
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# 0. PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
REPORT_DIR = SCRIPT_DIR / "reports"
for _d in (OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

ABLATION_CONFIGS = [
    "Full System",
    "Without Retrieval",
    "Without Persona",
    "Without SHAP",
    "Without Cognitive",
    "Risk Only",
]

DEFAULT_QUESTIONS = [
    "I'm feeling really behind in this course, what should I do?",
    "Can you explain the topic we covered this week?",
    "I don't understand today's material at all, I'm confused.",
    "How should I plan my revision before the next assessment?",
    "I haven't logged in for a while, where should I start again?",
]


# =============================================================================
# 1. BACKEND IMPORT (never modifies the imported files)
# =============================================================================

_CANDIDATE_PACKAGE_NAMES = ["services"]  # adjust here if your project's
                                          # backend package is named
                                          # differently (e.g. "app.services")

BACKEND_STATUS: Dict[str, Dict[str, Optional[str]]] = {
    "mentor_service": {"available": False, "error": None, "module_path": None},
    "context_builder": {"available": False, "error": None, "module_path": None},
    "prompt_builder": {"available": False, "error": None, "module_path": None},
    "retrieval": {"available": False, "error": None, "module_path": None},
    "confidence": {"available": False, "error": None, "module_path": None},
    "reasoning_layer": {"available": False, "error": None, "module_path": None},
    "supabase": {"available": False, "error": None, "module_path": None},
}

mentor_service = None
context_builder = None
prompt_builder = None
retrieval_mod = None
confidence_mod = None
reasoning_layer_mod = None
supabase_client = None


def _find_backend_roots() -> List[Path]:
    """
    Locate candidate project roots containing a `services/mentor_service.py`
    package, without assuming a fixed directory layout. Read-only search --
    never creates or edits anything.
    """
    candidates: List[Path] = []

    env_root = os.environ.get("BACKEND_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())

    p = SCRIPT_DIR
    for _ in range(8):
        for pkg_name in _CANDIDATE_PACKAGE_NAMES:
            if (p / pkg_name / "mentor_service.py").exists():
                candidates.append(p)
            if (p / "backend" / pkg_name / "mentor_service.py").exists():
                candidates.append(p / "backend")
        p = p.parent

    # de-dupe, preserve order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _add_backend_to_syspath() -> List[Path]:
    roots = _find_backend_roots()
    for r in roots:
        if str(r) not in sys.path:
            sys.path.insert(0, str(r))
    return roots


def _try_import_backend() -> None:
    global mentor_service, context_builder, prompt_builder
    global retrieval_mod, confidence_mod, reasoning_layer_mod, supabase_client

    roots = _add_backend_to_syspath()

    module_map = [
        ("mentor_service", "mentor_service"),
        ("context_builder", "context_builder"),
        ("prompt_builder", "prompt_builder"),
        ("retrieval", "retrieval"),
        ("confidence", "confidence"),
        ("reasoning_layer", "reasoning_layer"),
    ]

    for status_key, submodule in module_map:
        imported = None
        last_error = None
        for pkg_name in _CANDIDATE_PACKAGE_NAMES:
            full_name = f"{pkg_name}.{submodule}"
            try:
                imported = importlib.import_module(full_name)
                BACKEND_STATUS[status_key]["module_path"] = full_name
                break
            except Exception as e:  # noqa: BLE001 - we want to record any import failure
                last_error = f"{full_name}: {type(e).__name__}: {e}"
        if imported is not None:
            globals()[
                {
                    "mentor_service": "mentor_service",
                    "context_builder": "context_builder",
                    "prompt_builder": "prompt_builder",
                    "retrieval": "retrieval_mod",
                    "confidence": "confidence_mod",
                    "reasoning_layer": "reasoning_layer_mod",
                }[status_key]
            ] = imported
            BACKEND_STATUS[status_key]["available"] = True
        else:
            BACKEND_STATUS[status_key]["error"] = last_error or (
                f"No candidate package produced services.{submodule}; "
                f"searched roots: {[str(r) for r in roots]}"
            )

    try:
        from config.supabase_client import supabase as _supabase  # type: ignore
        supabase_client = _supabase
        BACKEND_STATUS["supabase"]["available"] = True
        BACKEND_STATUS["supabase"]["module_path"] = "config.supabase_client"
    except Exception as e:  # noqa: BLE001
        BACKEND_STATUS["supabase"]["error"] = f"{type(e).__name__}: {e}"


def _infer_model_name() -> str:
    """
    Reads the actual model string ask_mentor() calls, straight out of the
    real mentor_service module's source, instead of hardcoding a value that
    could silently drift out of sync with the real backend. Falls back to
    an explicit env override, then a documented default, if inspection
    fails for any reason (never silently guesses).
    """
    env_override = os.environ.get("OLLAMA_MODEL")
    if env_override:
        return env_override

    if mentor_service is not None:
        try:
            src = inspect.getsource(mentor_service.ask_mentor)
            m = re.search(r'model\s*=\s*["\']([^"\']+)["\']', src)
            if m:
                return m.group(1)
        except Exception:  # noqa: BLE001
            pass

    return "qwen2.5:3b"  # documented fallback; report states if this was used


def _check_ollama_reachable() -> (bool, Optional[str]):
    try:
        import ollama  # real dependency also used by mentor_service.py
        ollama.list()
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# 2. EVALUATION USER / QUESTION RESOLUTION (real users only)
# =============================================================================

def resolve_eval_user_ids(cli_user_ids: Optional[str]) -> List[str]:
    if cli_user_ids:
        return [u.strip() for u in cli_user_ids.split(",") if u.strip()]

    env_ids = os.environ.get("EVAL_USER_IDS", "")
    if env_ids.strip():
        return [u.strip() for u in env_ids.split(",") if u.strip()]

    if supabase_client is not None:
        try:
            resp = supabase_client.table("evaluation_users").select("user_id").execute()
            ids = [row["user_id"] for row in (resp.data or []) if row.get("user_id")]
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass

    return []


def resolve_eval_questions(cli_questions_file: Optional[str]) -> List[str]:
    path = cli_questions_file or os.environ.get("EVAL_QUESTIONS_FILE")
    if path and Path(path).exists():
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
        lines = [l for l in lines if l]
        if lines:
            return lines
    return list(DEFAULT_QUESTIONS)


# =============================================================================
# 3. ABLATION MASKING (removes real evidence, never fabricates replacements)
# =============================================================================

def mask_context(real_context: Any, config_name: str) -> Any:
    """
    Deep-copies the REAL StudentContext returned by build_student_context()
    and removes fields for the requested ablation arm. Never invents,
    substitutes, or hallucinates a replacement value -- fields are only
    ever set to None / False / empty, matching how the real downstream
    functions (prompt_builder, confidence) already interpret "no evidence".
    """
    masked = copy.deepcopy(real_context)

    if config_name == "Full System":
        return masked  # unused in practice -- Full System calls ask_mentor()
                        # directly rather than this masked-context path.

    if config_name == "Without Retrieval":
        # Retrieval itself is skipped at the call site (retrieved_chunks
        # forced to []); nothing to remove on the context object here.
        return masked

    if config_name == "Without Persona":
        # "Unknown Persona" is the SAME sentinel value confidence.py and
        # prompt_builder.py already use elsewhere to mean "no usable
        # persona" -- reused here only to mark this field as ablated for
        # this evaluation arm, never claimed to be real backend output.
        if hasattr(masked, "persona"):
            masked.persona = "Unknown Persona"
        if hasattr(masked, "intervention_style"):
            masked.intervention_style = None
        return masked

    if config_name == "Without SHAP":
        if hasattr(masked, "shap_available"):
            masked.shap_available = False
        if hasattr(masked, "shap_explanation"):
            masked.shap_explanation = None
        return masked

    if config_name == "Without Cognitive":
        if hasattr(masked, "cognitive_state_available"):
            masked.cognitive_state_available = False
        if hasattr(masked, "cognitive_state"):
            masked.cognitive_state = None
        return masked

    if config_name == "Risk Only":
        if hasattr(masked, "persona"):
            masked.persona = "Unknown Persona"
        if hasattr(masked, "intervention_style"):
            masked.intervention_style = None
        if hasattr(masked, "shap_available"):
            masked.shap_available = False
        if hasattr(masked, "shap_explanation"):
            masked.shap_explanation = None
        if hasattr(masked, "cognitive_state_available"):
            masked.cognitive_state_available = False
        if hasattr(masked, "cognitive_state"):
            masked.cognitive_state = None
        if hasattr(masked, "chat_history"):
            masked.chat_history = ""
        return masked

    raise ValueError(f"Unknown ablation config: {config_name}")


# =============================================================================
# 4. RECORD SCHEMA
# =============================================================================

@dataclass
class EvalRecord:
    user_id: str
    question: str
    config: str
    status: str  # "ok" | "context_unavailable" | "backend_unavailable" | "error"
    live_llm_used: Optional[bool] = None
    retrieval_available: Optional[bool] = None
    retrieved_chunk_count: Optional[int] = None
    retrieval_success: Optional[bool] = None
    persona: Optional[str] = None
    risk_score: Optional[int] = None
    risk_level: Optional[str] = None
    shap_available: Optional[bool] = None
    cognitive_state_available: Optional[bool] = None
    confidence_level: Optional[str] = None
    confidence_rationale: Optional[str] = None
    evidence_used: Optional[List[str]] = None
    parse_mode: Optional[str] = None  # "strict" | "lenient" | "fallback" | "n/a" | "not_observable"
    mentor_response_chars: Optional[int] = None
    student_recommendation_chars: Optional[int] = None
    educator_recommendation_chars: Optional[int] = None
    timings: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -------------------------------------------------------------------
    # NEW (response-quality extension) -- these fields store the SAME
    # values the script already computed (mr/sr/er text, masked_context
    # attributes, retrieved_chunks) but previously discarded after taking
    # len()/getattr() for the length-only fields above. No new backend
    # call, no new data-collection path: purely retaining more of an
    # already-obtained real value so the new quality metrics have
    # something real to score. Left as None (not fabricated) wherever the
    # calling branch does not actually have the value available -- see
    # inline comments at each assignment site.
    # -------------------------------------------------------------------
    mentor_response_text: Optional[str] = None
    student_recommendation_text: Optional[str] = None
    educator_recommendation_text: Optional[str] = None
    retrieved_chunk_topics: Optional[List[str]] = None
    retrieved_chunk_texts: Optional[List[str]] = None
    risk_reasons: Optional[List[str]] = None
    shap_explanation_text: Optional[str] = None
    cognitive_state_text: Optional[str] = None


# =============================================================================
# 5. ABLATION ARM ORCHESTRATION (calls real functions only, no reimplementation)
# =============================================================================

def run_ablation_call(
    user_id: str,
    question: str,
    config_name: str,
    real_context: Any,
    live_llm_active: bool,
    model_name: str,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    t_total0 = time.perf_counter()

    masked_context = mask_context(real_context, config_name)

    # --- Retrieval ---
    t0 = time.perf_counter()
    if config_name == "Without Retrieval":
        retrieved_chunks: List[Dict[str, str]] = []
    else:
        retriever = retrieval_mod.get_retriever()
        retrieved_chunks = retriever.retrieve(question, k=5)
    if hasattr(masked_context, "retrieved_chunks"):
        masked_context.retrieved_chunks = retrieved_chunks
    timings["retrieval"] = time.perf_counter() - t0

    # --- Prompt (real prompt_builder + real mentor_service formatting fn) ---
    t0 = time.perf_counter()
    system_prompt = prompt_builder.build_system_prompt(masked_context, retrieved_chunks)
    persona_for_fewshot = getattr(masked_context, "persona", "Unknown Persona")
    system_prompt += mentor_service._build_output_format_instructions(persona_for_fewshot)
    timings["prompt_build"] = time.perf_counter() - t0

    # --- Confidence (real, evidence-only rubric) ---
    t0 = time.perf_counter()
    confidence_result = confidence_mod.assess_confidence(
        masked_context, retrieved_chunks_found=len(retrieved_chunks) > 0
    )
    timings["confidence"] = time.perf_counter() - t0

    # --- LLM (only if live inference is actually active) ---
    raw_text: Optional[str] = None
    parse_mode = "n/a"
    if live_llm_active:
        t0 = time.perf_counter()
        import ollama  # same real dependency mentor_service.py uses
        response = ollama.chat(
            model=model_name,
            options={"temperature": 0.5, "top_p": 0.9, "num_predict": 480},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        raw_text = response["message"]["content"].strip()
        timings["llm"] = time.perf_counter() - t0
    else:
        timings["llm"] = 0.0

    parsed = {"mentor_response": None, "student_recommendation": None, "educator_recommendation": None}
    if raw_text is not None:
        # Observability only: check which of mentor_service's own real
        # regexes matched, purely to report parse_mode honestly. The
        # actual parsing/fallback content still comes entirely from
        # mentor_service._parse_structured_response() itself.
        if mentor_service._STRICT_PATTERN.search(raw_text):
            parse_mode = "strict"
        elif mentor_service._LENIENT_PATTERN.search(raw_text):
            parse_mode = "lenient"
        else:
            parse_mode = "fallback"
        parsed = mentor_service._parse_structured_response(raw_text, masked_context)

    timings["total"] = time.perf_counter() - t_total0

    return {
        "masked_context": masked_context,
        "retrieved_chunks": retrieved_chunks,
        "confidence_result": confidence_result,
        "parsed": parsed,
        "parse_mode": parse_mode,
        "timings": timings,
        "raw_text": raw_text,
        "system_prompt_len": len(system_prompt),
    }


# =============================================================================
# 6. TOP-LEVEL PER-(user, question, config) EVALUATION
# =============================================================================

def backend_ready_for_ablation() -> bool:
    required = ["mentor_service", "context_builder", "prompt_builder", "retrieval", "confidence"]
    return all(BACKEND_STATUS[k]["available"] for k in required)


def evaluate_one(
    user_id: str,
    question: str,
    config_name: str,
    live_llm_active: bool,
    model_name: str,
) -> EvalRecord:
    rec = EvalRecord(user_id=user_id, question=question, config=config_name, status="pending")

    if not backend_ready_for_ablation():
        rec.status = "backend_unavailable"
        rec.error = (
            "One or more required backend modules could not be imported "
            "(see outputs/backend_status.json); no metric was evaluated "
            "or estimated for this record."
        )
        return rec

    try:
        if config_name == "Full System":
            rec.live_llm_used = live_llm_active
            if not live_llm_active:
                rec.status = "error"
                rec.error = (
                    "Live LLM inference is not active (USE_LIVE_LLM=False or "
                    "Ollama unreachable). ask_mentor() unconditionally calls "
                    "the LLM internally and has no offline/deterministic "
                    "path of its own, so the Full System configuration "
                    "cannot be honestly evaluated in this run. See the "
                    "ablation-arm records for offline-evaluable metrics."
                )
                return rec

            result = mentor_service.ask_mentor(user_id, question)
            if result.get("status") != "success":
                rec.status = "error"
                rec.error = result.get("message") or "ask_mentor() returned a non-success status."
                return rec

            rec.status = "ok"
            rec.persona = result.get("persona")
            rec.risk_score = result.get("risk_score")
            rec.risk_level = result.get("risk_level")
            rec.shap_available = result.get("shap_available")
            rec.cognitive_state_available = result.get("cognitive_state_available")
            rec.confidence_level = result.get("confidence")
            rec.confidence_rationale = result.get("confidence_rationale")
            rec.evidence_used = result.get("evidence_used")
            rec.parse_mode = "not_observable"  # ask_mentor()'s public return
                                                # dict does not expose which
                                                # regex matched; not guessed.
            mr = result.get("mentor_response") or ""
            sr = result.get("student_recommendation") or ""
            er = result.get("educator_recommendation") or ""
            rec.mentor_response_chars = len(mr) if mr else None
            rec.student_recommendation_chars = len(sr) if sr else None
            rec.educator_recommendation_chars = len(er) if er else None
            # ask_mentor()'s return schema does not expose retrieved-chunk
            # count; left unobserved rather than inferred.
            rec.retrieval_available = None
            rec.retrieved_chunk_count = None
            rec.retrieval_success = None
            # NEW: retain the actual text ask_mentor() already returned
            # (previously only its length was kept). ask_mentor()'s public
            # return dict does not expose retrieved_chunks, risk_reasons,
            # shap_explanation, or cognitive_state content, so those stay
            # None here -- not guessed -- and any quality metric that
            # needs them will report "Not Evaluated" for this record.
            rec.mentor_response_text = mr or None
            rec.student_recommendation_text = sr or None
            rec.educator_recommendation_text = er or None
            return rec

        # ---- Ablation arms ----
        try:
            real_context = context_builder.build_student_context(user_id)
        except Exception as e:  # noqa: BLE001
            rec.status = "context_unavailable"
            rec.error = f"build_student_context() failed: {type(e).__name__}: {e}"
            return rec

        if real_context is None:
            rec.status = "context_unavailable"
            rec.error = "build_student_context() returned None for this user_id."
            return rec

        rec.live_llm_used = live_llm_active
        out = run_ablation_call(user_id, question, config_name, real_context, live_llm_active, model_name)

        rec.status = "ok"
        mc = out["masked_context"]
        rec.persona = getattr(mc, "persona", None)
        rec.risk_score = getattr(mc, "risk_score", None)
        rec.risk_level = getattr(mc, "risk_level", None)
        rec.shap_available = getattr(mc, "shap_available", None)
        rec.cognitive_state_available = getattr(mc, "cognitive_state_available", None)
        rec.retrieval_available = config_name != "Without Retrieval"
        rec.retrieved_chunk_count = len(out["retrieved_chunks"])
        rec.retrieval_success = len(out["retrieved_chunks"]) > 0
        cr = out["confidence_result"]
        rec.confidence_level = getattr(cr, "level", None)
        rec.confidence_rationale = getattr(cr, "rationale", None)
        rec.evidence_used = getattr(cr, "evidence_used", None)
        rec.parse_mode = out["parse_mode"]
        mr = out["parsed"].get("mentor_response")
        sr = out["parsed"].get("student_recommendation")
        er = out["parsed"].get("educator_recommendation")
        rec.mentor_response_chars = len(mr) if mr else None
        rec.student_recommendation_chars = len(sr) if sr else None
        rec.educator_recommendation_chars = len(er) if er else None
        rec.timings = out["timings"]
        # NEW: retain the actual text/evidence the ablation arm already
        # produced (previously only lengths/booleans were kept). All
        # values below come from objects the script already built
        # (masked_context, out["retrieved_chunks"], out["parsed"]) --
        # nothing new is fetched or computed here.
        rec.mentor_response_text = mr or None
        rec.student_recommendation_text = sr or None
        rec.educator_recommendation_text = er or None
        rec.retrieved_chunk_topics = (
            [c.get("topic") for c in out["retrieved_chunks"] if c.get("topic")]
            or None
        )
        rec.retrieved_chunk_texts = (
            [c.get("chunk") for c in out["retrieved_chunks"] if c.get("chunk")]
            or None
        )
        raw_risk_reasons = getattr(mc, "risk_reasons", None)
        rec.risk_reasons = list(raw_risk_reasons) if raw_risk_reasons else None
        raw_shap = getattr(mc, "shap_explanation", None)
        rec.shap_explanation_text = str(raw_shap) if raw_shap else None
        raw_cog = getattr(mc, "cognitive_state", None)
        rec.cognitive_state_text = str(raw_cog) if raw_cog else None
        return rec

    except Exception as e:  # noqa: BLE001
        rec.status = "error"
        rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"
        return rec


# =============================================================================
# 7. AGGREGATION (only over observed values; every stat traceable to records)
# =============================================================================

def aggregate(records: List[EvalRecord]) -> Dict[str, Any]:
    by_config: Dict[str, List[EvalRecord]] = {c: [] for c in ABLATION_CONFIGS}
    for r in records:
        by_config.setdefault(r.config, []).append(r)

    summary: Dict[str, Any] = {"per_config": {}, "overall": {}}

    total_ok = sum(1 for r in records if r.status == "ok")
    total = len(records)
    summary["overall"] = {
        "total_records": total,
        "ok": total_ok,
        "status_counts": _count_by(records, lambda r: r.status),
    }

    for cfg, recs in by_config.items():
        if not recs:
            summary["per_config"][cfg] = {"note": "no records for this configuration in this run"}
            continue

        ok_recs = [r for r in recs if r.status == "ok"]
        live_ok_recs = [r for r in ok_recs if r.live_llm_used]

        retrieval_recs = [r for r in ok_recs if r.retrieval_success is not None]
        confidence_recs = [r for r in ok_recs if r.confidence_level is not None]
        response_len_recs = [r for r in live_ok_recs if r.mentor_response_chars is not None]

        summary["per_config"][cfg] = {
            "n_records": len(recs),
            "status_counts": _count_by(recs, lambda r: r.status),
            "retrieval_coverage": (
                round(sum(1 for r in retrieval_recs if r.retrieval_success) / len(retrieval_recs), 4)
                if retrieval_recs else None
            ),
            "n_retrieval_observed": len(retrieval_recs),
            "confidence_distribution": _count_by(confidence_recs, lambda r: r.confidence_level) or None,
            "parse_mode_distribution": _count_by(
                [r for r in ok_recs if r.parse_mode not in (None, "n/a", "not_observable")],
                lambda r: r.parse_mode,
            ) or None,
            "mean_mentor_response_chars": (
                round(sum(r.mentor_response_chars for r in response_len_recs) / len(response_len_recs), 1)
                if response_len_recs else None
            ),
            "n_response_length_observed": len(response_len_recs),
            "live_llm_used_count": sum(1 for r in recs if r.live_llm_used),
            "offline_count": sum(1 for r in recs if r.live_llm_used is False),
        }

    return summary


def _count_by(records: List[EvalRecord], keyfn) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        k = keyfn(r)
        if k is None:
            continue
        counts[k] = counts.get(k, 0) + 1
    return counts


# =============================================================================
# 7B. RESPONSE QUALITY METRICS (new -- deterministic / documented-rubric)
# =============================================================================
"""
This section adds research-grade response-quality evaluation on top of the
system diagnostics already collected above (retrieval coverage, confidence
distribution, parse-mode, response length). It does NOT change how any
record was produced, does NOT call any backend module differently, and does
NOT modify mentor_service / prompt_builder / reasoning_layer / confidence /
retrieval / context_builder / persona_service / risk_service in any way --
it only reads fields already captured on EvalRecord (see the "NEW" fields
added above and their population sites in evaluate_one()).

Every metric below is one of two kinds, and each is labelled as such in the
report:

  (a) DETERMINISTIC -- a pure function of data already on the record
      (e.g. "did the response text contain the persona's own name",
      "what fraction of retrieved-chunk tokens reappear in the response").
      These are exact, reproducible, and require no judgement calls.

  (b) DOCUMENTED HEURISTIC RUBRIC -- still deterministic code, but the
      rubric itself encodes a design choice (e.g. which keywords count as
      "urgency language"). These are explicitly labelled as heuristic
      proxies for the underlying construct, not as ground truth measures
      of it. The rubric is stated in full below so it can be audited or
      replaced.

If the fields a metric needs are not present on a record (e.g. offline run
with no LLM text, or ask_mentor()'s return schema not exposing
retrieved_chunks/shap_explanation/risk_reasons/cognitive_state), that
metric is reported as the literal string "Not Evaluated" for that record --
never estimated, never defaulted to a number.
"""

import math
import re as _re
from collections import Counter

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "for", "and", "or", "with", "your", "you", "it", "this",
    "that", "as", "at", "by", "from", "will", "can", "should", "how", "what",
    "do", "does", "did", "not", "no", "so", "if", "we", "our", "i", "my",
    "me", "have", "has", "had", "into", "about", "than", "then", "there",
    "their", "them", "up", "out", "over", "also", "such", "which", "these",
    "those",
}


def _tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    tokens = _re.findall(r"[a-zA-Z']+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def _token_set(text: Optional[str]) -> set:
    return set(_tokenize(text))


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else None


def _recall(needle_tokens: set, haystack_tokens: set) -> Optional[float]:
    """Fraction of needle_tokens that also appear in haystack_tokens."""
    if not needle_tokens:
        return None
    if not haystack_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


NOT_EVALUATED = "Not Evaluated"

# --- Rubric keyword sets (documented heuristics; edit here, not in code) ---
_URGENCY_WORDS = {"urgent", "priority", "important", "immediately", "right now",
                   "first", "focus on", "top priority", "critical"}
_STEADY_WORDS = {"keep up", "continue", "steady", "maintain", "great job",
                  "consistent", "stay on track", "well done"}
_ACTION_VERBS = {"review", "practice", "schedule", "revisit", "try", "set",
                  "track", "break down", "reach out", "reconnect", "plan",
                  "organize", "attend", "message", "email", "book", "start",
                  "revise", "summarize", "watch", "re-watch", "attempt"}
_TIME_MARKERS = {"today", "tomorrow", "daily", "weekly", "this week",
                  "this weekend", "by friday", "next class", "hour", "minutes"}
_EXPLANATION_MARKERS = {"because", "due to", "driven by", "as a result of",
                          "this is because", "the reason", "given that"}
_GENERIC_FALLBACK_PHRASES = [
    # Known boilerplate observed in the legacy mentor_rag_engine.py
    # prototype's hardcoded response template -- if a "recommendation"
    # matches this near-verbatim, it is almost certainly a template
    # fallback rather than a personalized recommendation.
    "focus on understanding the core concepts step by step",
    "revise slowly and practice related examples",
    "you are improving consistently",
]


@dataclass
class QualityScore:
    user_id: str
    question: str
    config: str
    record_status: str
    evidence_fusion_correctness: Any = NOT_EVALUATED
    evidence_fusion_detail: Optional[str] = None
    priority_focus_correctness: Any = NOT_EVALUATED
    priority_focus_detail: Optional[str] = None
    recommendation_quality: Any = NOT_EVALUATED
    recommendation_quality_detail: Optional[str] = None
    persona_consistency: Any = NOT_EVALUATED  # filled in a second pass (needs peers)
    educational_usefulness: Any = NOT_EVALUATED
    grounding_shap: Any = NOT_EVALUATED
    grounding_risk: Any = NOT_EVALUATED
    grounding_cognitive: Any = NOT_EVALUATED
    grounding_overall: Any = NOT_EVALUATED
    # recommendation_diversity and tone_differentiation are population-level
    # (per config), not per-record -- see aggregate_quality().


def _score_evidence_fusion(r: EvalRecord) -> (Any, Optional[str]):
    """
    DETERMINISTIC (keyword-presence proxy). For each evidence category that
    was actually available on this record (per EvalRecord's own booleans /
    values -- not re-derived), check whether the response text contains a
    topical marker for that category. Score = (#referenced) / (#available).
    A category the record did not have is simply excluded from the
    denominator, never penalized as "missing."
    NOTE: this measures topical co-occurrence, not semantic correctness of
    the fusion (e.g. it cannot tell if the persona was used *correctly*,
    only whether it was mentioned/reflected at all). This limitation is
    disclosed in the report.
    """
    if r.status != "ok" or not r.mentor_response_text:
        return NOT_EVALUATED, "no response text on this record"

    resp_tokens = _token_set(r.mentor_response_text)
    available, referenced = [], []

    if r.risk_level:
        available.append("risk")
        if resp_tokens & {"risk", "risky"} or "at-risk" in r.mentor_response_text.lower():
            referenced.append("risk")
    if r.persona and r.persona != "Unknown Persona":
        available.append("persona")
        persona_tokens = _token_set(r.persona)
        if persona_tokens & resp_tokens:
            referenced.append("persona")
    if r.retrieval_available and r.retrieved_chunk_topics:
        available.append("retrieval")
        topic_tokens = set()
        for t in r.retrieved_chunk_topics:
            topic_tokens |= _token_set(t)
        if topic_tokens & resp_tokens:
            referenced.append("retrieval")
    elif r.retrieval_available is None:
        pass  # not exposed for this config (e.g. Full System) -- excluded, not penalized
    if r.shap_available:
        available.append("shap")
        if resp_tokens & _EXPLANATION_MARKERS or any(
            m in r.mentor_response_text.lower() for m in _EXPLANATION_MARKERS
        ):
            referenced.append("shap")
    if r.cognitive_state_available:
        available.append("cognitive")
        if resp_tokens & {"focus", "confusion", "engagement", "attention", "overload"}:
            referenced.append("cognitive")

    if not available:
        return NOT_EVALUATED, "no evidence categories were available on this record"

    score = round(len(referenced) / len(available), 3)
    detail = f"referenced {referenced} of available {available}"
    return score, detail


def _score_priority_focus(r: EvalRecord) -> (Any, Optional[str]):
    """
    DOCUMENTED HEURISTIC RUBRIC. High-risk records are expected to open
    with urgency/priority language; Low-risk records are expected to use
    steady/maintenance language. Medium risk is not scored (ambiguous
    expected tone) -- reported "Not Evaluated" rather than guessed.
    """
    if r.status != "ok" or not r.mentor_response_text:
        return NOT_EVALUATED, "no response text on this record"
    if r.risk_level not in ("High", "Low"):
        return NOT_EVALUATED, f"risk_level={r.risk_level!r} not scored by this rubric"

    text = r.mentor_response_text.lower()
    has_urgency = any(w in text for w in _URGENCY_WORDS)
    has_steady = any(w in text for w in _STEADY_WORDS)

    if r.risk_level == "High":
        match = has_urgency and not has_steady
    else:  # Low
        match = has_steady and not has_urgency

    return (1 if match else 0), (
        f"risk_level={r.risk_level}, urgency_language={has_urgency}, "
        f"steady_language={has_steady}"
    )


def _score_recommendation_quality(r: EvalRecord) -> (Any, Optional[str]):
    """
    DOCUMENTED HEURISTIC RUBRIC, 0-3 points, on student_recommendation_text:
      +1 specific enough (>= 20 characters)
      +1 contains at least one actionable verb (see _ACTION_VERBS)
      +1 contains a time marker (see _TIME_MARKERS)
    Capped at 0 (regardless of the above) if the text matches a known
    generic fallback phrase from the legacy prototype template, since that
    indicates a template default rather than a real recommendation.
    """
    text = r.student_recommendation_text
    if r.status != "ok" or not text:
        return NOT_EVALUATED, "no student_recommendation text on this record"

    lower = text.lower()
    if any(p in lower for p in _GENERIC_FALLBACK_PHRASES):
        return 0, "matched a known generic/fallback template phrase"

    points = 0
    reasons = []
    if len(text.strip()) >= 20:
        points += 1
        reasons.append("specific-length")
    if any(v in lower for v in _ACTION_VERBS):
        points += 1
        reasons.append("actionable-verb")
    if any(tm in lower for tm in _TIME_MARKERS):
        points += 1
        reasons.append("time-marker")

    return points, f"{points}/3 ({', '.join(reasons) if reasons else 'none matched'})"


def _score_educational_usefulness(r: EvalRecord) -> (Any, Optional[str]):
    """
    DETERMINISTIC recall metric: what fraction of the salient (non-stopword)
    tokens in the retrieved learning-material chunks reappear in the
    mentor's response. Proxy for "did the response actually use the
    retrieved material" rather than ignoring it in favor of generic advice.
    Only computable for ablation arms where retrieved_chunk_texts was
    captured (Full System does not expose this -- see EvalRecord notes).
    """
    if r.status != "ok" or not r.mentor_response_text:
        return NOT_EVALUATED, "no response text on this record"
    if not r.retrieved_chunk_texts:
        return NOT_EVALUATED, "retrieved chunk text not available for this config"

    chunk_tokens: set = set()
    for c in r.retrieved_chunk_texts:
        chunk_tokens |= _token_set(c)
    resp_tokens = _token_set(r.mentor_response_text)
    score = _recall(chunk_tokens, resp_tokens)
    if score is None:
        return NOT_EVALUATED, "retrieved chunk text had no scoreable tokens"
    return round(score, 3), f"{len(chunk_tokens & resp_tokens)}/{len(chunk_tokens)} chunk tokens reused"


def _score_grounding(r: EvalRecord) -> Dict[str, Any]:
    """
    DETERMINISTIC recall metrics, one per evidence source, only computed
    where the source text was actually captured on the record (ablation
    arms; Full System's return schema does not expose these, so all three
    -- and the overall -- report "Not Evaluated" for Full System records).
    """
    out = {"shap": NOT_EVALUATED, "risk": NOT_EVALUATED, "cognitive": NOT_EVALUATED, "overall": NOT_EVALUATED}
    if r.status != "ok" or not r.mentor_response_text:
        return out

    resp_tokens = _token_set(r.mentor_response_text)
    scored = []

    if r.shap_explanation_text:
        s = _recall(_token_set(r.shap_explanation_text), resp_tokens)
        if s is not None:
            out["shap"] = round(s, 3)
            scored.append(s)

    if r.risk_reasons:
        s = _recall(_token_set(" ".join(r.risk_reasons)), resp_tokens)
        if s is not None:
            out["risk"] = round(s, 3)
            scored.append(s)

    if r.cognitive_state_text:
        s = _recall(_token_set(r.cognitive_state_text), resp_tokens)
        if s is not None:
            out["cognitive"] = round(s, 3)
            scored.append(s)

    if scored:
        out["overall"] = round(sum(scored) / len(scored), 3)

    return out


def evaluate_response_quality(records: List[EvalRecord]) -> List[QualityScore]:
    """Per-record quality scoring (first pass). Persona consistency, which
    needs peer records, is filled in by _fill_persona_consistency() after."""
    scores: List[QualityScore] = []
    for r in records:
        qs = QualityScore(
            user_id=r.user_id, question=r.question, config=r.config,
            record_status=r.status,
        )
        qs.evidence_fusion_correctness, qs.evidence_fusion_detail = _score_evidence_fusion(r)
        qs.priority_focus_correctness, qs.priority_focus_detail = _score_priority_focus(r)
        qs.recommendation_quality, qs.recommendation_quality_detail = _score_recommendation_quality(r)
        qs.educational_usefulness, _ = _score_educational_usefulness(r)
        g = _score_grounding(r)
        qs.grounding_shap, qs.grounding_risk = g["shap"], g["risk"]
        qs.grounding_cognitive, qs.grounding_overall = g["cognitive"], g["overall"]
        scores.append(qs)

    _fill_persona_consistency(records, scores)
    return scores


def _fill_persona_consistency(records: List[EvalRecord], scores: List[QualityScore]) -> None:
    """
    DETERMINISTIC. Persona Consistency = mean pairwise token-Jaccard
    similarity between mentor_response_text values sharing the same
    (config, persona), across different questions for that persona. A
    persona that reliably produces thematically-similar guidance scores
    high; a persona whose responses are indistinguishable from any other
    persona's would need the separate Tone Differentiation check (§ below,
    population-level) to reveal that -- this metric alone only checks
    self-consistency, not distinctiveness from other personas.
    Requires >= 2 comparable responses for the same (config, persona);
    otherwise "Not Evaluated".
    """
    groups: Dict[tuple, List[int]] = {}
    for i, r in enumerate(records):
        if r.status == "ok" and r.persona and r.persona != "Unknown Persona" and r.mentor_response_text:
            groups.setdefault((r.config, r.persona), []).append(i)

    for (_cfg, _persona), idxs in groups.items():
        if len(idxs) < 2:
            for i in idxs:
                scores[i].persona_consistency = NOT_EVALUATED
            continue
        token_sets = [_token_set(records[i].mentor_response_text) for i in idxs]
        sims = []
        for a in range(len(token_sets)):
            for b in range(a + 1, len(token_sets)):
                j = _jaccard(token_sets[a], token_sets[b])
                if j is not None:
                    sims.append(j)
        group_score = round(sum(sims) / len(sims), 3) if sims else NOT_EVALUATED
        for i in idxs:
            scores[i].persona_consistency = group_score


def _stats(values: List[float]) -> Dict[str, Any]:
    """Mean / std / 95% CI (normal approximation) over observed numeric
    values only. Returns 'Not Evaluated' if fewer than 2 values."""
    if len(values) < 2:
        return {
            "n": len(values), "mean": (values[0] if values else NOT_EVALUATED),
            "std": NOT_EVALUATED, "ci95_low": NOT_EVALUATED, "ci95_high": NOT_EVALUATED,
            "note": "fewer than 2 observations; std/CI not computable" if len(values) < 2 else None,
        }
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    # 1.96 normal-approximation multiplier; noted as approximate for small n.
    margin = 1.96 * se
    return {
        "n": n, "mean": round(mean, 4), "std": round(std, 4),
        "ci95_low": round(mean - margin, 4), "ci95_high": round(mean + margin, 4),
        "note": "95% CI uses a normal approximation; treat as approximate for n < 30" if n < 30 else None,
    }


def _numeric_values(scores: List[QualityScore], field_name: str, config: Optional[str] = None) -> List[float]:
    out = []
    for s in scores:
        if config is not None and s.config != config:
            continue
        v = getattr(s, field_name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out


_PER_RECORD_METRIC_FIELDS = [
    "evidence_fusion_correctness", "priority_focus_correctness",
    "recommendation_quality", "persona_consistency",
    "educational_usefulness", "grounding_shap", "grounding_risk",
    "grounding_cognitive", "grounding_overall",
]


def _paired_comparison(
    scores: List[QualityScore], field_name: str, baseline_config: str, other_config: str
) -> Dict[str, Any]:
    """
    Paired comparison of a per-record numeric metric between two
    configurations, paired by (user_id, question) so each pair reflects
    the same underlying student+question with only the evidence config
    differing. Only pairs where BOTH sides have a real numeric score are
    used (no imputation). Reports the mean/std of the paired differences
    (other - baseline) plus a paired t-statistic if scipy is available.
    """
    by_key_baseline = {(s.user_id, s.question): s for s in scores if s.config == baseline_config}
    by_key_other = {(s.user_id, s.question): s for s in scores if s.config == other_config}

    diffs = []
    for key, base_s in by_key_baseline.items():
        other_s = by_key_other.get(key)
        if other_s is None:
            continue
        bv, ov = getattr(base_s, field_name), getattr(other_s, field_name)
        if isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
            diffs.append(ov - bv)

    if len(diffs) < 2:
        return {"n_pairs": len(diffs), "mean_diff": NOT_EVALUATED, "note": "fewer than 2 paired observations"}

    n = len(diffs)
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var)
    result = {"n_pairs": n, "mean_diff": round(mean_diff, 4), "std_diff": round(std_diff, 4)}
    try:
        from scipy import stats as _scipy_stats  # optional dependency
        t_stat, p_value = _scipy_stats.ttest_rel(
            [getattr(by_key_other[k], field_name) for k in by_key_baseline if k in by_key_other
             and isinstance(getattr(by_key_baseline[k], field_name), (int, float))
             and isinstance(getattr(by_key_other[k], field_name), (int, float))],
            [getattr(by_key_baseline[k], field_name) for k in by_key_baseline if k in by_key_other
             and isinstance(getattr(by_key_baseline[k], field_name), (int, float))
             and isinstance(getattr(by_key_other[k], field_name), (int, float))],
        )
        result["t_stat"] = round(float(t_stat), 4)
        result["p_value"] = round(float(p_value), 4)
    except Exception as e:  # noqa: BLE001 -- scipy optional; never fabricate a p-value
        result["t_stat"] = NOT_EVALUATED
        result["p_value"] = NOT_EVALUATED
        result["stat_note"] = f"scipy unavailable or comparison failed ({type(e).__name__}); reporting mean/std of paired differences only"

    # -------------------------------------------------------------------
    # NEW: assumption-aware test selection (paired t-test vs Wilcoxon
    # signed-rank) + effect size. Purely additive -- does not change
    # n_pairs / mean_diff / std_diff / t_stat / p_value above, which
    # remain exactly as before for backward compatibility. Uses the SAME
    # `diffs` already computed above; no new data collection.
    # -------------------------------------------------------------------
    result.update(_recommended_significance_test(diffs))
    return result


def _recommended_significance_test(diffs: List[float]) -> Dict[str, Any]:
    """
    Chooses between a paired t-test and a Wilcoxon signed-rank test based
    on a Shapiro-Wilk normality check of the paired differences (when
    scipy is available and n is large enough for Shapiro to run, i.e.
    n >= 3). If the differences look non-normal, or n is too small for a
    reliable normality check, Wilcoxon (the assumption-light alternative)
    is preferred, per the study's stated preference order. Also reports
    an effect size:
      - Cohen's d_z (mean_diff / std_diff) for the t-test path.
      - Matched-pairs rank-biserial correlation r for the Wilcoxon path
        (r = Z / sqrt(N), with Z approximated from the reported p-value
        and the sign of the median difference -- noted as approximate).
    Never estimates a p-value or effect size when scipy is unavailable or
    the test cannot be run (e.g. all differences are zero) -- reports
    "Not Evaluated" instead.
    """
    n = len(diffs)
    out: Dict[str, Any] = {
        "recommended_test": NOT_EVALUATED,
        "recommended_p_value": NOT_EVALUATED,
        "recommended_statistic": NOT_EVALUATED,
        "normality_p_value": NOT_EVALUATED,
        "effect_size": NOT_EVALUATED,
        "effect_size_type": NOT_EVALUATED,
        "sample_size": n,
    }
    if n < 2:
        out["stat_note"] = "fewer than 2 paired observations; no significance test performed"
        return out

    try:
        from scipy import stats as _scipy_stats  # optional dependency
    except Exception as e:  # noqa: BLE001
        out["stat_note"] = f"scipy unavailable ({type(e).__name__}); no significance test performed"
        return out

    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_diff = math.sqrt(var)

    is_normal: Optional[bool] = None
    if n >= 3:
        try:
            _, shapiro_p = _scipy_stats.shapiro(diffs)
            out["normality_p_value"] = round(float(shapiro_p), 4)
            is_normal = shapiro_p > 0.05
        except Exception:  # noqa: BLE001
            is_normal = None  # unknown -- fall through to the Wilcoxon default below

    prefer_ttest = bool(is_normal)  # None or False both mean "prefer Wilcoxon"

    if prefer_ttest and std_diff > 0:
        t_stat, p_value = _scipy_stats.ttest_1samp(diffs, popmean=0.0)
        out["recommended_test"] = "paired t-test"
        out["recommended_statistic"] = round(float(t_stat), 4)
        out["recommended_p_value"] = round(float(p_value), 4)
        out["effect_size"] = round(mean_diff / std_diff, 4)
        out["effect_size_type"] = "Cohen's d_z (mean paired diff / std of paired diffs)"
        return out

    # Wilcoxon signed-rank path (non-normal, unknown normality, or the
    # t-test path was unavailable). Wilcoxon requires at least one
    # non-zero difference.
    nonzero_diffs = [d for d in diffs if d != 0]
    if len(nonzero_diffs) < 1:
        out["recommended_test"] = "Wilcoxon signed-rank"
        out["stat_note"] = "all paired differences are exactly zero; Wilcoxon not computable"
        return out
    try:
        w_stat, w_p = _scipy_stats.wilcoxon(diffs)
        out["recommended_test"] = "Wilcoxon signed-rank"
        out["recommended_statistic"] = round(float(w_stat), 4)
        out["recommended_p_value"] = round(float(w_p), 4)
        # Approximate rank-biserial effect size via a normal approximation
        # of the Wilcoxon p-value, signed by the direction of mean_diff.
        # This is a documented approximation, not an exact z-statistic
        # from scipy (scipy's default 'auto' method does not always
        # expose one) -- reported as approximate in the report text.
        try:
            z_approx = _scipy_stats.norm.ppf(1 - (w_p / 2))
            if mean_diff < 0:
                z_approx = -z_approx
            out["effect_size"] = round(float(z_approx) / math.sqrt(len(nonzero_diffs)), 4)
            out["effect_size_type"] = "matched-pairs rank-biserial r (approximate, from normal-approx Z)"
        except Exception:  # noqa: BLE001
            out["effect_size"] = NOT_EVALUATED
            out["effect_size_type"] = NOT_EVALUATED
    except Exception as e:  # noqa: BLE001
        out["recommended_test"] = "Wilcoxon signed-rank"
        out["stat_note"] = f"Wilcoxon test failed ({type(e).__name__}: {e})"
    return out


def _recommendation_diversity(scores: List[QualityScore], records: List[EvalRecord], config: str) -> Dict[str, Any]:
    """
    Population-level (per config), DETERMINISTIC. Two measures over all
    student_recommendation_text values produced under this config:
      - unique_ratio: distinct texts / total texts (exact-match diversity)
      - mean_pairwise_dissimilarity: 1 - Jaccard, averaged over all pairs
        (captures near-duplicate phrasing that unique_ratio would miss)
    """
    texts = [r.student_recommendation_text for r in records
             if r.config == config and r.status == "ok" and r.student_recommendation_text]
    if len(texts) < 2:
        return {"n": len(texts), "unique_ratio": NOT_EVALUATED, "mean_pairwise_dissimilarity": NOT_EVALUATED,
                "note": "fewer than 2 recommendation texts for this config"}

    unique_ratio = round(len(set(texts)) / len(texts), 3)
    token_sets = [_token_set(t) for t in texts]
    dissims = []
    for a in range(len(token_sets)):
        for b in range(a + 1, len(token_sets)):
            j = _jaccard(token_sets[a], token_sets[b])
            if j is not None:
                dissims.append(1 - j)
    mean_dissim = round(sum(dissims) / len(dissims), 3) if dissims else NOT_EVALUATED
    return {"n": len(texts), "unique_ratio": unique_ratio, "mean_pairwise_dissimilarity": mean_dissim}


def _tone_differentiation(records: List[EvalRecord], config: str) -> Dict[str, Any]:
    """
    Population-level (per config), DETERMINISTIC. Compares mean token-
    Jaccard similarity of mentor_response_text WITHIN the same persona
    vs. ACROSS different personas. tone_differentiation =
    within_persona_similarity - across_persona_similarity; positive means
    same-persona responses are more textually similar to each other than
    to other personas' responses (i.e. tone actually differs by persona).
    Requires >= 2 personas each with >= 1 response in this config.
    """
    by_persona: Dict[str, List[set]] = {}
    for r in records:
        if r.config == config and r.status == "ok" and r.persona and r.persona != "Unknown Persona" and r.mentor_response_text:
            by_persona.setdefault(r.persona, []).append(_token_set(r.mentor_response_text))

    personas = [p for p, v in by_persona.items() if v]
    if len(personas) < 2:
        return {"within_persona_similarity": NOT_EVALUATED, "across_persona_similarity": NOT_EVALUATED,
                "tone_differentiation": NOT_EVALUATED, "note": "fewer than 2 personas observed in this config"}

    within_sims = []
    for p in personas:
        sets = by_persona[p]
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                j = _jaccard(sets[a], sets[b])
                if j is not None:
                    within_sims.append(j)

    across_sims = []
    for i in range(len(personas)):
        for j_ in range(i + 1, len(personas)):
            for sa in by_persona[personas[i]]:
                for sb in by_persona[personas[j_]]:
                    j = _jaccard(sa, sb)
                    if j is not None:
                        across_sims.append(j)

    within = round(sum(within_sims) / len(within_sims), 3) if within_sims else NOT_EVALUATED
    across = round(sum(across_sims) / len(across_sims), 3) if across_sims else NOT_EVALUATED
    diff = round(within - across, 3) if isinstance(within, float) and isinstance(across, float) else NOT_EVALUATED
    return {"within_persona_similarity": within, "across_persona_similarity": across,
            "tone_differentiation": diff, "n_personas_observed": len(personas)}


def aggregate_quality(records: List[EvalRecord], scores: List[QualityScore]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_config": {}, "paired_vs_full_system": {}}

    for cfg in ABLATION_CONFIGS:
        cfg_out = {}
        for field_name in _PER_RECORD_METRIC_FIELDS:
            vals = _numeric_values(scores, field_name, config=cfg)
            not_eval_count = sum(
                1 for s in scores if s.config == cfg and getattr(s, field_name) == NOT_EVALUATED
            )
            cfg_out[field_name] = {**_stats(vals), "n_not_evaluated": not_eval_count}
        cfg_out["recommendation_diversity"] = _recommendation_diversity(scores, records, cfg)
        cfg_out["tone_differentiation"] = _tone_differentiation(records, cfg)
        out["per_config"][cfg] = cfg_out

    if "Full System" in ABLATION_CONFIGS:
        for cfg in ABLATION_CONFIGS:
            if cfg == "Full System":
                continue
            out["paired_vs_full_system"][cfg] = {
                field_name: _paired_comparison(scores, field_name, "Full System", cfg)
                for field_name in _PER_RECORD_METRIC_FIELDS
            }

    return out


# =============================================================================
# 7C. CORRELATION ANALYSIS (new -- deterministic, over already-collected data)
# =============================================================================
"""
Computes correlations between quantitative signals already present on the
same record: the confidence level `assess_confidence()` actually assigned
(mapped to an ordinal 1-3 scale: Low=1, Medium=2, High=3, since that is
literally the ordering the rubric in confidence.py already encodes -- not
a new numeric value invented for this analysis) and the quality metrics
already scored in section 7B. Uses Spearman's rank correlation, which is
the appropriate choice for an ordinal variable (confidence level) paired
with a bounded/interval-ish metric, per the study's stated preference for
"Pearson or Spearman as appropriate." Never computed, and reported
"Not Evaluated", when fewer than 3 paired real observations exist.
"""

_CONFIDENCE_ORDINAL = {"Low": 1, "Medium": 2, "High": 3}


def _correlation_analysis(records: List[EvalRecord], scores: List[QualityScore]) -> Dict[str, Any]:
    """
    `records` and `scores` are the SAME length and same order (scores is
    produced by evaluate_response_quality() by iterating `records` once,
    one QualityScore per EvalRecord) -- paired by index here, not by any
    new lookup or join logic.
    """
    pairs_by_metric = {
        "recommendation_quality": [],
        "educational_usefulness": [],
        "grounding_overall": [],
    }
    for r, s in zip(records, scores):
        if r.status != "ok" or r.confidence_level not in _CONFIDENCE_ORDINAL:
            continue
        conf_val = _CONFIDENCE_ORDINAL[r.confidence_level]
        for field_name in pairs_by_metric:
            v = getattr(s, field_name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                pairs_by_metric[field_name].append((conf_val, float(v)))

    metric_labels = {
        "recommendation_quality": "Confidence vs Recommendation Quality",
        "educational_usefulness": "Confidence vs Educational Usefulness",
        "grounding_overall": "Confidence vs Grounding Recall",
    }

    out: Dict[str, Any] = {}
    for field_name, label in metric_labels.items():
        pairs = pairs_by_metric[field_name]
        n = len(pairs)
        if n < 3:
            out[field_name] = {
                "label": label, "n": n, "coefficient": NOT_EVALUATED,
                "p_value": NOT_EVALUATED, "method": NOT_EVALUATED,
                "note": "fewer than 3 paired observations with real confidence and metric values",
            }
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        try:
            from scipy import stats as _scipy_stats  # optional dependency
            coef, p_value = _scipy_stats.spearmanr(xs, ys)
            if math.isnan(coef) or math.isnan(p_value):
                out[field_name] = {
                    "label": label, "n": n, "coefficient": NOT_EVALUATED,
                    "p_value": NOT_EVALUATED, "method": NOT_EVALUATED,
                    "note": "one or both variables were constant across observed pairs; correlation undefined",
                }
            else:
                out[field_name] = {
                    "label": label, "n": n, "coefficient": round(float(coef), 4),
                    "p_value": round(float(p_value), 4), "method": "Spearman",
                }
        except Exception as e:  # noqa: BLE001 -- scipy optional; never fabricate a coefficient
            out[field_name] = {
                "label": label, "n": n, "coefficient": NOT_EVALUATED,
                "p_value": NOT_EVALUATED, "method": NOT_EVALUATED,
                "note": f"scipy unavailable or computation failed ({type(e).__name__}: {e})",
            }
    return out


# =============================================================================
# 7D. ERROR ANALYSIS (new -- computed strictly from observed records)
# =============================================================================

def _error_analysis(records: List[EvalRecord]) -> Dict[str, Any]:
    """
    All values below are counted directly from `records` -- nothing
    estimated. Reports "Not Evaluated" for any sub-metric with no
    observations to count.
    """
    out: Dict[str, Any] = {"per_config": {}, "overall": {}}

    by_config: Dict[str, List[EvalRecord]] = {c: [] for c in ABLATION_CONFIGS}
    for r in records:
        by_config.setdefault(r.config, []).append(r)

    worst_cfg, worst_rate = None, -1.0
    for cfg in ABLATION_CONFIGS:
        recs = by_config.get(cfg, [])
        if not recs:
            out["per_config"][cfg] = {"note": "no records for this configuration in this run"}
            continue
        n = len(recs)
        n_fail = sum(1 for r in recs if r.status != "ok")
        failure_rate = round(n_fail / n, 4)

        parse_observed = [r for r in recs if r.parse_mode in ("strict", "lenient", "fallback")]
        n_fallback = sum(1 for r in parse_observed if r.parse_mode == "fallback")
        n_compliant = sum(1 for r in parse_observed if r.parse_mode in ("strict", "lenient"))
        fallback_freq = round(n_fallback / len(parse_observed), 4) if parse_observed else NOT_EVALUATED
        compliance_rate = round(n_compliant / len(parse_observed), 4) if parse_observed else NOT_EVALUATED

        len_recs = [r.mentor_response_chars for r in recs if r.mentor_response_chars is not None]
        avg_len = round(sum(len_recs) / len(len_recs), 1) if len_recs else NOT_EVALUATED

        conf_recs = [r.confidence_level for r in recs if r.confidence_level is not None]
        conf_dist = _count_by(recs, lambda r: r.confidence_level) or NOT_EVALUATED

        out["per_config"][cfg] = {
            "n_records": n,
            "n_failed": n_fail,
            "failure_rate": failure_rate,
            "parser_fallback_frequency": fallback_freq,
            "n_parse_mode_observed": len(parse_observed),
            "structured_response_compliance_rate": compliance_rate,
            "average_response_length_chars": avg_len,
            "n_response_length_observed": len(len_recs),
            "confidence_distribution": conf_dist,
        }
        if recs and failure_rate > worst_rate:
            worst_rate, worst_cfg = failure_rate, cfg

    out["overall"]["highest_failure_rate_config"] = worst_cfg if worst_cfg is not None else NOT_EVALUATED
    out["overall"]["highest_failure_rate_value"] = worst_rate if worst_cfg is not None else NOT_EVALUATED

    all_parse_observed = [r for r in records if r.parse_mode in ("strict", "lenient", "fallback")]
    if all_parse_observed:
        n_fb = sum(1 for r in all_parse_observed if r.parse_mode == "fallback")
        n_ok = sum(1 for r in all_parse_observed if r.parse_mode in ("strict", "lenient"))
        out["overall"]["parser_fallback_frequency"] = round(n_fb / len(all_parse_observed), 4)
        out["overall"]["structured_response_compliance_rate"] = round(n_ok / len(all_parse_observed), 4)
    else:
        out["overall"]["parser_fallback_frequency"] = NOT_EVALUATED
        out["overall"]["structured_response_compliance_rate"] = NOT_EVALUATED

    out["overall"]["confidence_distribution"] = _count_by(records, lambda r: r.confidence_level) or NOT_EVALUATED
    return out


# =============================================================================
# 7E. REPRODUCIBILITY SUMMARY (new -- reuses values already computed at runtime)
# =============================================================================

def _compute_backend_sha256() -> Any:
    """
    Hashes the concatenated source of the real, already-imported backend
    modules this script actually called (in a fixed, deterministic order),
    so the report can state exactly which backend code produced its
    numbers. Reads source via `inspect.getsource()` on the modules already
    imported by `_try_import_backend()` -- does not re-read or guess file
    paths, and reports "Not Evaluated" for any module that did not import.
    """
    mods = [
        ("mentor_service", mentor_service),
        ("context_builder", context_builder),
        ("prompt_builder", prompt_builder),
        ("retrieval", retrieval_mod),
        ("confidence", confidence_mod),
        ("reasoning_layer", reasoning_layer_mod),
    ]
    import hashlib
    hasher = hashlib.sha256()
    any_hashed = False
    missing = []
    for name, mod in mods:
        if mod is None:
            missing.append(name)
            continue
        try:
            src = inspect.getsource(mod)
            hasher.update(name.encode("utf-8"))
            hasher.update(src.encode("utf-8"))
            any_hashed = True
        except Exception:  # noqa: BLE001
            missing.append(name)
    if not any_hashed:
        return NOT_EVALUATED, missing
    return hasher.hexdigest(), missing


def build_reproducibility_summary(
    records: List[EvalRecord],
    user_ids: List[str],
    questions: List[str],
    model_name: str,
    live_llm_active: bool,
) -> Dict[str, Any]:
    sha, missing = _compute_backend_sha256()
    return {
        "backend_sha256": sha,
        "backend_sha256_missing_modules": missing or None,
        "ollama_model": model_name if live_llm_active else NOT_EVALUATED,
        "live_llm_active": live_llm_active,
        "number_of_evaluated_users": len(user_ids),
        "number_of_evaluated_questions": len(questions),
        "number_of_ablation_configs": len(ABLATION_CONFIGS),
        "total_evaluations": len(records),
        "timestamp_utc": RUN_TIMESTAMP,
    }


# =============================================================================
# 7F. AUTOMATIC KEY FINDINGS (new -- computed from summary/quality_summary,
# never hard-coded)
# =============================================================================

def build_key_findings(
    summary: Dict[str, Any],
    quality_summary: Dict[str, Any],
    error_analysis: Dict[str, Any],
    correlations: Dict[str, Any],
) -> Dict[str, Any]:
    findings: Dict[str, Any] = {}

    pvf = quality_summary.get("paired_vs_full_system", {}) if quality_summary else {}

    # -- Which configuration degraded performance the most? --
    # Average the (arm - Full System) mean_diff across all metrics that
    # have a real value for that config, then take the most negative.
    config_avg_diff: Dict[str, float] = {}
    per_config_metric_diffs: Dict[str, Dict[str, float]] = {}
    for cfg, metrics in pvf.items():
        diffs = []
        metric_diffs = {}
        for field_name, pc in metrics.items():
            md = pc.get("mean_diff")
            if isinstance(md, (int, float)):
                diffs.append(md)
                metric_diffs[field_name] = md
        if diffs:
            config_avg_diff[cfg] = sum(diffs) / len(diffs)
            per_config_metric_diffs[cfg] = metric_diffs

    if config_avg_diff:
        worst_cfg = min(config_avg_diff, key=config_avg_diff.get)
        findings["most_degrading_configuration"] = {
            "configuration": worst_cfg,
            "mean_diff_averaged_across_metrics": round(config_avg_diff[worst_cfg], 4),
            "note": "Most negative mean (arm - Full System) averaged across all quality metrics with a real paired value.",
        }
    else:
        findings["most_degrading_configuration"] = NOT_EVALUATED

    # -- Which component had the greatest influence? --
    # The single (config, metric) pair with the largest |mean_diff|.
    best_pair, best_mag = None, -1.0
    for cfg, metric_diffs in per_config_metric_diffs.items():
        for field_name, md in metric_diffs.items():
            if abs(md) > best_mag:
                best_mag, best_pair = abs(md), (cfg, field_name, md)
    if best_pair:
        cfg, field_name, md = best_pair
        component = cfg.replace("Without ", "").replace("Risk Only", "Persona+SHAP+Cognitive+History")
        findings["greatest_influence_component"] = {
            "component": component, "configuration": cfg, "metric": field_name,
            "mean_diff": round(md, 4),
        }
    else:
        findings["greatest_influence_component"] = NOT_EVALUATED

    # -- Which metrics remained stable / were most sensitive? --
    metric_spreads: Dict[str, List[float]] = {}
    for cfg, metric_diffs in per_config_metric_diffs.items():
        for field_name, md in metric_diffs.items():
            metric_spreads.setdefault(field_name, []).append(md)

    if metric_spreads:
        metric_max_abs = {m: max(abs(v) for v in vs) for m, vs in metric_spreads.items()}
        most_stable = min(metric_max_abs, key=metric_max_abs.get)
        most_sensitive = max(metric_max_abs, key=metric_max_abs.get)
        findings["most_stable_metric"] = {
            "metric": most_stable, "max_abs_mean_diff_across_configs": round(metric_max_abs[most_stable], 4),
        }
        findings["most_sensitive_metric"] = {
            "metric": most_sensitive, "max_abs_mean_diff_across_configs": round(metric_max_abs[most_sensitive], 4),
        }
    else:
        findings["most_stable_metric"] = NOT_EVALUATED
        findings["most_sensitive_metric"] = NOT_EVALUATED

    # -- Highest failure-rate configuration (from error analysis) --
    findings["highest_failure_rate_configuration"] = {
        "configuration": error_analysis.get("overall", {}).get("highest_failure_rate_config", NOT_EVALUATED),
        "failure_rate": error_analysis.get("overall", {}).get("highest_failure_rate_value", NOT_EVALUATED),
    }

    # -- Strongest observed correlation --
    strongest = None
    for field_name, c in (correlations or {}).items():
        coef = c.get("coefficient")
        if isinstance(coef, (int, float)):
            if strongest is None or abs(coef) > abs(strongest[1]):
                strongest = (c.get("label", field_name), coef, c.get("p_value"), c.get("n"))
    findings["strongest_correlation"] = (
        {"pair": strongest[0], "coefficient": strongest[1], "p_value": strongest[2], "n": strongest[3]}
        if strongest else NOT_EVALUATED
    )

    return findings


# =============================================================================
# 8. FIGURES (matplotlib; only produced when real data exists)
# =============================================================================

def generate_figures(
    summary: Dict[str, Any],
    quality_summary: Optional[Dict[str, Any]] = None,
    _raw_quality_scores: Optional[List["QualityScore"]] = None,
) -> Dict[str, str]:
    """
    Returns a dict of {figure_name: status}, where status is either the
    saved file path or a human-readable reason the figure was omitted.
    Never draws a figure from data that isn't present in `summary`.

    `_raw_quality_scores` is optional and additive: when provided (the
    per-record list from evaluate_response_quality()), it enables the new
    box/violin distribution figure; existing figures are unaffected if
    it is omitted.
    """
    results: Dict[str, str] = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        reason = f"matplotlib unavailable ({type(e).__name__}: {e}); all figures omitted"
        for name in ("retrieval_coverage", "confidence_distribution", "mean_response_length",
                     "quality_metrics_by_config"):
            results[name] = reason
        return results

    per_config = summary.get("per_config", {})

    # --- Figure 1: retrieval coverage by configuration ---
    labels, values = [], []
    for cfg in ABLATION_CONFIGS:
        cov = per_config.get(cfg, {}).get("retrieval_coverage")
        if cov is not None:
            labels.append(cfg)
            values.append(cov)
    if labels:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_ylabel("Retrieval coverage (fraction with \u22651 chunk)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Retrieval coverage by ablation configuration (observed)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path = FIGURE_DIR / "retrieval_coverage.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        results["retrieval_coverage"] = str(out_path)
    else:
        results["retrieval_coverage"] = "no retrieval observations were recorded in this run"

    # --- Figure 2: confidence level distribution by configuration ---
    levels = ["High", "Medium", "Low"]
    configs_with_data = [
        cfg for cfg in ABLATION_CONFIGS
        if per_config.get(cfg, {}).get("confidence_distribution")
    ]
    if configs_with_data:
        fig, ax = plt.subplots(figsize=(9, 5))
        bottoms = [0] * len(configs_with_data)
        colors = {"High": "#55A868", "Medium": "#DD8452", "Low": "#C44E52"}
        for level in levels:
            counts = []
            for cfg in configs_with_data:
                dist = per_config[cfg]["confidence_distribution"] or {}
                counts.append(dist.get(level, 0))
            ax.bar(configs_with_data, counts, bottom=bottoms, label=level, color=colors[level])
            bottoms = [b + c for b, c in zip(bottoms, counts)]
        ax.set_ylabel("Number of observed responses")
        ax.set_title("Confidence-level distribution by ablation configuration (observed)")
        ax.legend(title="Confidence")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path = FIGURE_DIR / "confidence_distribution.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        results["confidence_distribution"] = str(out_path)
    else:
        results["confidence_distribution"] = "no confidence observations were recorded in this run"

    # --- Figure 3: mean mentor_response length by configuration (live-LLM only) ---
    labels, values = [], []
    for cfg in ABLATION_CONFIGS:
        v = per_config.get(cfg, {}).get("mean_mentor_response_chars")
        if v is not None:
            labels.append(cfg)
            values.append(v)
    if labels:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#8172B2")
        ax.set_ylabel("Mean mentor_response length (characters)")
        ax.set_title("Mean live-LLM response length by ablation configuration (observed)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path = FIGURE_DIR / "mean_response_length.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        results["mean_response_length"] = str(out_path)
    else:
        results["mean_response_length"] = "no live-LLM responses were recorded in this run"

    # --- Figure 4: response-quality metric means by configuration (new) ---
    if quality_summary:
        qpc = quality_summary.get("per_config", {})
        metric_labels = {
            "evidence_fusion_correctness": "Evidence Fusion",
            "priority_focus_correctness": "Priority Focus",
            "recommendation_quality": "Recommendation Quality (0-3)",
            "persona_consistency": "Persona Consistency",
            "educational_usefulness": "Educational Usefulness",
            "grounding_overall": "Grounding (overall)",
        }
        configs_present = [c for c in ABLATION_CONFIGS if c in qpc]
        any_data = False
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (field_name, label) in zip(axes.flat, metric_labels.items()):
            labels, values = [], []
            for cfg in configs_present:
                m = qpc[cfg].get(field_name, {})
                if isinstance(m.get("mean"), (int, float)):
                    labels.append(cfg)
                    values.append(m["mean"])
            if labels:
                any_data = True
                ax.bar(labels, values, color="#4C72B0")
                ax.set_title(label, fontsize=10)
                ax.tick_params(axis="x", rotation=30, labelsize=7)
            else:
                ax.set_title(label + " (no data)", fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        if any_data:
            out_path = FIGURE_DIR / "quality_metrics_by_config.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results["quality_metrics_by_config"] = str(out_path)
        else:
            plt.close(fig)
            results["quality_metrics_by_config"] = "no response-quality metrics were observed in this run"
    else:
        results["quality_metrics_by_config"] = "quality_summary not provided to generate_figures()"

    # --- Figure 5 (new): 95% CI error-bar chart for deterministic + heuristic metrics ---
    if quality_summary:
        qpc = quality_summary.get("per_config", {})
        ci_metrics = [
            ("recommendation_quality", "Recommendation Quality (heuristic, 0-3)"),
            ("evidence_fusion_correctness", "Evidence Fusion Correctness (heuristic, 0-1)"),
            ("educational_usefulness", "Educational Usefulness (deterministic, 0-1)"),
            ("grounding_overall", "Grounding Recall (deterministic, 0-1)"),
        ]
        any_ci_data = False
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (field_name, label) in zip(axes.flat, ci_metrics):
            labels, means, err_low, err_high = [], [], [], []
            for cfg in ABLATION_CONFIGS:
                m = qpc.get(cfg, {}).get(field_name, {})
                mean = m.get("mean")
                ci_lo, ci_hi = m.get("ci95_low"), m.get("ci95_high")
                if isinstance(mean, (int, float)) and isinstance(ci_lo, (int, float)) and isinstance(ci_hi, (int, float)):
                    labels.append(cfg)
                    means.append(mean)
                    err_low.append(mean - ci_lo)
                    err_high.append(ci_hi - mean)
            if labels:
                any_ci_data = True
                x = range(len(labels))
                ax.errorbar(x, means, yerr=[err_low, err_high], fmt="o", capsize=4, color="#4C72B0")
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
                ax.set_title(label, fontsize=10)
            else:
                ax.set_title(label + " (no CI-eligible data)", fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        if any_ci_data:
            out_path = FIGURE_DIR / "quality_metrics_ci_error_bars.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results["quality_metrics_ci_error_bars"] = str(out_path)
        else:
            plt.close(fig)
            results["quality_metrics_ci_error_bars"] = "no metrics had >=2 observations (CI not computable) in this run"
    else:
        results["quality_metrics_ci_error_bars"] = "quality_summary not provided to generate_figures()"

    # --- Figure 6 (new): box/violin plots of per-record metric distributions ---
    # Uses the raw per-record quality scores (passed separately, since summary
    # only holds aggregated stats) when available.
    if _raw_quality_scores:
        box_metrics = [
            ("recommendation_quality", "Recommendation Quality (0-3)"),
            ("evidence_fusion_correctness", "Evidence Fusion Correctness (0-1)"),
            ("educational_usefulness", "Educational Usefulness (0-1)"),
            ("grounding_overall", "Grounding Recall (0-1)"),
        ]
        any_box_data = False
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (field_name, label) in zip(axes.flat, box_metrics):
            data, labels = [], []
            for cfg in ABLATION_CONFIGS:
                vals = [
                    getattr(s, field_name) for s in _raw_quality_scores
                    if s.config == cfg and isinstance(getattr(s, field_name), (int, float))
                    and not isinstance(getattr(s, field_name), bool)
                ]
                if vals:
                    data.append(vals)
                    labels.append(cfg)
            if data:
                any_box_data = True
                parts = ax.violinplot(data, showmeans=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#8172B2")
                    pc.set_alpha(0.4)
                ax.boxplot(data, positions=range(1, len(data) + 1), widths=0.15, showfliers=False)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
                ax.set_title(label, fontsize=10)
            else:
                ax.set_title(label + " (no per-record data)", fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        if any_box_data:
            out_path = FIGURE_DIR / "quality_metrics_box_violin.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results["quality_metrics_box_violin"] = str(out_path)
        else:
            plt.close(fig)
            results["quality_metrics_box_violin"] = "no per-record quality scores were observed in this run"
    else:
        results["quality_metrics_box_violin"] = "raw quality scores not provided to generate_figures()"

    return results


# =============================================================================
# 9. REPORT GENERATION
# =============================================================================

def generate_report(
    records: List[EvalRecord],
    summary: Dict[str, Any],
    figure_status: Dict[str, str],
    user_ids: List[str],
    questions: List[str],
    live_llm_active: bool,
    ollama_error: Optional[str],
    model_name: str,
    quality_summary: Optional[Dict[str, Any]] = None,
    error_analysis: Optional[Dict[str, Any]] = None,
    correlations: Optional[Dict[str, Any]] = None,
    key_findings: Optional[Dict[str, Any]] = None,
    reproducibility: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = []
    a = lines.append

    a("# EduGuard-AI -- Real End-to-End RAG / Mentor Ablation Study\n")
    a(f"_Generated: {RUN_TIMESTAMP}_\n")

    a("## 0. Automatic Key Findings\n")
    a(
        "Computed automatically from the aggregated results below (\u00a76b, \u00a76e, \u00a76f) -- "
        "nothing in this section is hard-coded. Any entry reading `Not Evaluated` means the "
        "underlying comparison did not have enough real paired data in this run.\n"
    )
    if not key_findings:
        a("_Key findings were not computed for this run (no records / no quality summary)._\n")
    else:
        kf = key_findings
        mdc = kf.get("most_degrading_configuration")
        if isinstance(mdc, dict):
            a(f"- **Configuration that degraded performance the most:** `{mdc['configuration']}` "
              f"(mean diff averaged across metrics vs Full System = {mdc['mean_diff_averaged_across_metrics']}).")
        else:
            a("- **Configuration that degraded performance the most:** Not Evaluated.")
        gic = kf.get("greatest_influence_component")
        if isinstance(gic, dict):
            a(f"- **Component with the greatest single-metric influence:** `{gic['component']}` "
              f"(largest effect on `{gic['metric']}`, mean diff = {gic['mean_diff']}).")
        else:
            a("- **Component with the greatest single-metric influence:** Not Evaluated.")
        msm = kf.get("most_stable_metric")
        if isinstance(msm, dict):
            a(f"- **Most stable metric across ablations:** `{msm['metric']}` "
              f"(max |mean diff| across configs = {msm['max_abs_mean_diff_across_configs']}).")
        else:
            a("- **Most stable metric across ablations:** Not Evaluated.")
        mse = kf.get("most_sensitive_metric")
        if isinstance(mse, dict):
            a(f"- **Most sensitive metric across ablations:** `{mse['metric']}` "
              f"(max |mean diff| across configs = {mse['max_abs_mean_diff_across_configs']}).")
        else:
            a("- **Most sensitive metric across ablations:** Not Evaluated.")
        hfc = kf.get("highest_failure_rate_configuration")
        if isinstance(hfc, dict) and hfc.get("configuration") not in (None, NOT_EVALUATED):
            a(f"- **Configuration with the highest failure rate:** `{hfc['configuration']}` "
              f"(failure rate = {hfc['failure_rate']}).")
        else:
            a("- **Configuration with the highest failure rate:** Not Evaluated.")
        sc = kf.get("strongest_correlation")
        if isinstance(sc, dict):
            a(f"- **Strongest observed correlation:** {sc['pair']} (Spearman r = {sc['coefficient']}, "
              f"p = {sc['p_value']}, n = {sc['n']}).")
        else:
            a("- **Strongest observed correlation:** Not Evaluated.")

    a("## 1. What Was Actually Evaluated\n")
    a(
        "This report reflects live calls into the real AI Mentor backend "
        "(`mentor_service.ask_mentor`, `context_builder.build_student_context`, "
        "`prompt_builder.build_system_prompt`, `retrieval.get_retriever`, "
        "`confidence.assess_confidence`) for the user IDs and questions "
        "listed below. No student context, persona, risk score, SHAP "
        "explanation, cognitive state, retrieval result, or mentor response "
        "in this report was fabricated. Any metric that could not be "
        "observed is explicitly marked as such rather than estimated.\n"
    )

    a("### 1.1 Backend Modules Used\n")
    a("| Module | Imported | Path used | Error (if any) |")
    a("|---|---|---|---|")
    for key, status in BACKEND_STATUS.items():
        a(
            f"| `{key}` | {'yes' if status['available'] else 'NO'} | "
            f"{status['module_path'] or '-'} | {status['error'] or '-'} |"
        )

    a("\n### 1.2 Live Inference Status\n")
    a(f"- `USE_LIVE_LLM` requested: **{live_llm_active if ollama_error is None else 'requested'}**")
    a(f"- Ollama reachability probe (`ollama.list()`): **{'reachable' if live_llm_active else 'unreachable'}**")
    if ollama_error:
        a(f"- Ollama probe error: `{ollama_error}`")
    a(f"- Model name used for ablation-arm LLM calls: `{model_name}` "
      f"(read from `mentor_service.ask_mentor`'s own source where possible)")
    if not live_llm_active:
        a(
            "\n**This run used OFFLINE / deterministic evaluation only.** "
            "No LLM text was generated. The `Full System` configuration -- "
            "which has no deterministic path inside `ask_mentor()` itself -- "
            "is reported as `error` for every record in this run for that "
            "reason (see \u00a72.1). Ablation-arm records still contain real, "
            "live-measured retrieval and confidence data with the LLM call "
            "skipped.\n"
        )

    a("\n### 1.3 Users and Questions\n")
    user_ids_text = ", ".join(user_ids) if user_ids else "NONE -- see \u00a77"
    a(f"- User IDs evaluated ({len(user_ids)}): `{user_ids_text}`")
    a(f"- Questions evaluated ({len(questions)}):")
    for q in questions:
        a(f"  - {q}")

    a("\n## 2. Methodology\n")
    a("### 2.1 Full System Configuration\n")
    a(
        "Calls the real, unmodified `ask_mentor(user_id, question)` exactly "
        "as production traffic would, including its write to the "
        "`mentor_history` table. This is the only configuration that is a "
        "literal call into the real orchestrator end to end.\n"
    )
    a("### 2.2 Ablation Arms\n")
    a(
        "`ask_mentor()` builds its own `StudentContext` internally and has "
        "no parameter for injecting a modified one, so it cannot itself "
        "run masked-evidence ablations. Each ablation arm therefore: (1) "
        "calls the real `build_student_context(user_id)`, (2) deep-copies "
        "the real result and removes (never substitutes) the evidence "
        "field(s) that arm is meant to ablate, (3) replays the same "
        "sequence of real functions `ask_mentor()` calls internally -- "
        "`retrieval.get_retriever()`, `prompt_builder.build_system_prompt()`, "
        "`mentor_service._build_output_format_instructions()`, "
        "`confidence.assess_confidence()`, and (if live) the real "
        "`ollama.chat()` call followed by `mentor_service._parse_structured_response()`. "
        "No business logic is reimplemented anywhere in this script; every "
        "step above calls the real function from the real module. To avoid "
        "writing ablation-arm traffic into the production `mentor_history` "
        "table, these arms do not persist history (unlike Full System, "
        "\u00a72.1).\n"
    )
    a("Fields removed per arm (masking only, never replaced with a fabricated value):\n")
    a("- **Without Retrieval**: `retrieved_chunks` forced to `[]` before prompt construction.")
    a("- **Without Persona**: `persona` set to `\"Unknown Persona\"` (the system's own existing "
      "sentinel for \"no usable persona\", reused here only as a masking marker for this ablation "
      "arm, not claimed as real backend output); `intervention_style` set to `None`.")
    a("- **Without SHAP**: `shap_available=False`, `shap_explanation=None`.")
    a("- **Without Cognitive**: `cognitive_state_available=False`, `cognitive_state=None`.")
    a("- **Risk Only**: all of the above combined, plus `chat_history=\"\"`.")

    a("\n## 3. Aggregate Metrics (observed only)\n")
    a(f"- Total records attempted: **{summary['overall']['total_records']}**")
    a(f"- Records with status `ok`: **{summary['overall']['ok']}**")
    a("- Status breakdown: " + ", ".join(
        f"`{k}`={v}" for k, v in summary["overall"]["status_counts"].items()
    ) if summary["overall"]["status_counts"] else "- Status breakdown: (no records)")

    a("\n| Configuration | n | Status counts | Retrieval coverage | Confidence dist. | Parse-mode dist. | Mean response chars |")
    a("|---|---|---|---|---|---|---|")
    for cfg in ABLATION_CONFIGS:
        pc = summary["per_config"].get(cfg, {})
        if "note" in pc:
            a(f"| {cfg} | 0 | - | - | - | - | - |  _{pc['note']}_")
            continue
        a(
            f"| {cfg} | {pc['n_records']} | "
            f"{pc['status_counts']} | "
            f"{pc['retrieval_coverage'] if pc['retrieval_coverage'] is not None else 'unavailable'} | "
            f"{pc['confidence_distribution'] if pc['confidence_distribution'] else 'unavailable'} | "
            f"{pc['parse_mode_distribution'] if pc['parse_mode_distribution'] else 'unavailable'} | "
            f"{pc['mean_mentor_response_chars'] if pc['mean_mentor_response_chars'] is not None else 'unavailable'} |"
        )

    a("\n## 4. Figures\n")
    for name, status in figure_status.items():
        if status.endswith(".png"):
            rel = os.path.relpath(status, REPORT_DIR)
            a(f"### {name.replace('_', ' ').title()}\n")
            a(f"![{name}]({rel})\n")
        else:
            a(f"### {name.replace('_', ' ').title()}\n")
            a(f"_Omitted: {status}_\n")

    a("\n## 5. Explicit Limitations\n")
    limitations = []
    if not live_llm_active:
        limitations.append(
            "Live LLM inference was not active for this run; all mentor-response-derived "
            "metrics (response length, parse-mode) are unavailable, and `Full System` could "
            "not be evaluated at all (see \u00a71.2, \u00a72.1)."
        )
    for key, status in BACKEND_STATUS.items():
        if not status["available"]:
            limitations.append(f"Backend module `{key}` was not importable ({status['error']}); "
                                f"any metric depending on it is marked unavailable, not estimated.")
    if not user_ids:
        limitations.append(
            "No real evaluation user IDs were resolved (no --user-ids, no EVAL_USER_IDS, no "
            "`evaluation_users` table rows found); no per-student evaluation was performed."
        )
    if not limitations:
        limitations.append("None identified beyond what is already noted per-metric above.")
    for l in limitations:
        a(f"- {l}")

    a("\n## 6b. Response Quality Metrics (new)\n")
    a(
        "These metrics evaluate the actual generated text (mentor_response / "
        "student_recommendation), not just system diagnostics. Per the study's "
        "restructuring requirement, they are now split into two clearly labeled "
        "subsections -- **Deterministic Metrics**, computed directly from "
        "measurable data with no design-choice rubric involved, and **Heuristic "
        "Metrics**, which are rule-based evaluation rubrics (the rubric itself "
        "encodes a design choice, e.g. which keywords count as \"urgency "
        "language\") and should be read as documented proxies rather than "
        "objective benchmark scores. Full rubric definitions are in code "
        "comments, `rag_ablation.py` §7B. None are LLM-judged or hand-scored, "
        "and none are estimated where data was missing -- those cells read "
        "`Not Evaluated`.\n"
    )
    if not quality_summary:
        a("_Response-quality metrics were not computed for this run._\n")
    else:
        qpc = quality_summary.get("per_config", {})
        deterministic_rows = [
            ("educational_usefulness", "Educational Usefulness", "0-1, recall of retrieved-chunk tokens in the response"),
            ("persona_consistency", "Persona Consistency", "0-1, within-persona response similarity"),
            ("grounding_overall", "Grounding Recall", "0-1, mean recall across available evidence sources (SHAP/risk/cognitive)"),
        ]
        heuristic_rows = [
            ("recommendation_quality", "Recommendation Quality", "0-3, rule-based rubric (specificity + actionable verb + time marker)"),
            ("evidence_fusion_correctness", "Evidence Fusion Correctness", "0-1, keyword-presence rubric for referencing available evidence"),
            ("priority_focus_correctness", "Priority Focus Correctness", "0/1, rubric matching urgency/steady language to risk level"),
        ]
        _grounding_subrows = [
            ("grounding_shap", "SHAP grounding", "0-1, deterministic recall"),
            ("grounding_risk", "Risk-reasons grounding", "0-1, deterministic recall"),
            ("grounding_cognitive", "Cognitive-state grounding", "0-1, deterministic recall"),
        ]
        # Full field list retained for 6b.3 (stats) / 6b.4 (paired comparison)
        # below so that output is a superset of the previous report, not a
        # subset -- nothing that was reported before is dropped.
        metric_rows = deterministic_rows + heuristic_rows + _grounding_subrows

        def _render_metric_table(rows):
            a("| Metric | Scale | " + " | ".join(ABLATION_CONFIGS) + " |")
            a("|---|---|" + "---|" * len(ABLATION_CONFIGS))
            for field_name, label, scale in rows:
                cells = []
                for cfg in ABLATION_CONFIGS:
                    m = qpc.get(cfg, {}).get(field_name, {})
                    mean = m.get("mean", NOT_EVALUATED)
                    n = m.get("n", 0)
                    ne = m.get("n_not_evaluated", 0)
                    if mean == NOT_EVALUATED or mean is None:
                        cells.append(f"Not Evaluated (n_not_evaluated={ne})")
                    else:
                        cells.append(f"{mean} (n={n}, not_eval={ne})")
                a(f"| {label} | {scale} | " + " | ".join(cells) + " |")

        a("### 6b.1 Deterministic Metrics (mean scores by configuration, observed only)\n")
        a("These are computed directly from measurable data -- no rubric design choice is involved.\n")
        _render_metric_table(deterministic_rows)
        a("\nAlso deterministic, but population-level rather than per-record (see 6b.2): "
          "**Recommendation Diversity** and **Tone Differentiation**.\n")

        a("\n### 6b.1b Heuristic Metrics (mean scores by configuration, observed only)\n")
        a("These are rule-based evaluation rubrics -- documented proxies for the underlying "
          "construct, not objective benchmark measurements.\n")
        _render_metric_table(heuristic_rows)

        a("\n### 6b.1c Grounding Recall detail (deterministic sub-scores by evidence source)\n")
        a("Grounding Recall (6b.1) is the mean of whichever of these three are available per record.\n")
        _render_metric_table(_grounding_subrows)

        a("\n### 6b.2 Population-level metrics by configuration (deterministic)\n")
        a("| Configuration | Recommendation Diversity (unique ratio) | Mean pairwise dissimilarity | Tone differentiation (within - across persona similarity) |")
        a("|---|---|---|---|")
        for cfg in ABLATION_CONFIGS:
            div = qpc.get(cfg, {}).get("recommendation_diversity", {})
            tone = qpc.get(cfg, {}).get("tone_differentiation", {})
            a(
                f"| {cfg} | {div.get('unique_ratio', NOT_EVALUATED)} "
                f"(n={div.get('n', 0)}) | {div.get('mean_pairwise_dissimilarity', NOT_EVALUATED)} | "
                f"{tone.get('tone_differentiation', NOT_EVALUATED)} "
                f"(within={tone.get('within_persona_similarity', NOT_EVALUATED)}, "
                f"across={tone.get('across_persona_similarity', NOT_EVALUATED)}) |"
            )

        a("\n### 6b.3 Statistical detail (mean, std, 95% CI; observed only)\n")
        a("_95% CIs use a normal approximation and are noted as approximate for n < 30. "
          "Cells with fewer than 2 observations report `Not Evaluated` for std/CI._\n")
        a("| Metric | Configuration | n | Mean | Std | 95% CI |")
        a("|---|---|---|---|---|---|")
        for field_name, label, _scale in metric_rows:
            for cfg in ABLATION_CONFIGS:
                m = qpc.get(cfg, {}).get(field_name, {})
                if not m or m.get("mean") in (None, NOT_EVALUATED):
                    continue
                ci = f"[{m.get('ci95_low', NOT_EVALUATED)}, {m.get('ci95_high', NOT_EVALUATED)}]" \
                    if m.get("ci95_low") != NOT_EVALUATED else NOT_EVALUATED
                a(f"| {label} | {cfg} | {m.get('n')} | {m.get('mean')} | {m.get('std', NOT_EVALUATED)} | {ci} |")

        a("\n### 6b.4 Paired comparison vs. Full System (same user+question, evidence config differs)\n")
        a(
            "Pairs are formed only where BOTH the Full System record and the ablation-arm record "
            "for the same `(user_id, question)` have a real numeric score for that metric -- no "
            "imputation. `p_value`/`t_stat` (paired t-test) require `scipy`; if unavailable, only "
            "the mean/std of paired differences is reported and `p_value` reads `Not Evaluated`. "
            "The `Recommended test` columns additionally choose between a paired t-test and a "
            "Wilcoxon signed-rank test per-comparison, based on a Shapiro-Wilk normality check of "
            "the paired differences (Wilcoxon is preferred whenever normality is rejected or "
            "cannot be checked, per the study's stated test preference), and report an effect size "
            "alongside it.\n"
        )
        pvf = quality_summary.get("paired_vs_full_system", {})
        if not pvf:
            a("_No ablation arms were compared (Full System not present or no other configs run)._\n")
        else:
            a("| Configuration | Metric | n pairs | Mean diff (arm - Full System) | Std diff | t-stat | p-value | "
              "Recommended test | Recommended p-value | Effect size |")
            a("|---|---|---|---|---|---|---|---|---|---|")
            for cfg, metrics in pvf.items():
                for field_name, label, _scale in metric_rows:
                    pc = metrics.get(field_name, {})
                    if pc.get("mean_diff") in (None, NOT_EVALUATED):
                        continue
                    eff = pc.get("effect_size", NOT_EVALUATED)
                    eff_type = pc.get("effect_size_type", NOT_EVALUATED)
                    eff_str = f"{eff} ({eff_type})" if eff != NOT_EVALUATED else NOT_EVALUATED
                    a(
                        f"| {cfg} | {label} | {pc.get('n_pairs')} | {pc.get('mean_diff')} | "
                        f"{pc.get('std_diff', NOT_EVALUATED)} | {pc.get('t_stat', NOT_EVALUATED)} | "
                        f"{pc.get('p_value', NOT_EVALUATED)} | {pc.get('recommended_test', NOT_EVALUATED)} | "
                        f"{pc.get('recommended_p_value', NOT_EVALUATED)} | {eff_str} |"
                    )

        a("\n### 6b.5 Discussion (automatic interpretation of the paired comparisons above)\n")
        a(
            "Interpretive sentences below are generated directly from the `mean_diff` values in "
            "6b.4 -- no explanation is invented beyond what those measured differences show. "
            "Percent change is relative to the Full System mean for that metric where that mean "
            "is a real (non-zero) observed value; otherwise only the raw mean difference is "
            "stated.\n"
        )
        if not pvf:
            a("_Not Evaluated -- no paired comparisons were available to interpret._\n")
        else:
            interpretations = []
            unchanged_by_config: Dict[str, List[str]] = {}
            for cfg, metrics in pvf.items():
                for field_name, label, _scale in metric_rows:
                    pc = metrics.get(field_name, {})
                    md = pc.get("mean_diff")
                    if not isinstance(md, (int, float)):
                        continue
                    if md == 0:
                        unchanged_by_config.setdefault(cfg, []).append(label)
                        continue
                    full_mean = qpc.get("Full System", {}).get(field_name, {}).get("mean")
                    direction = "increased" if md > 0 else ("decreased" if md < 0 else "did not change")
                    p_val = pc.get("recommended_p_value", pc.get("p_value", NOT_EVALUATED))
                    sig_clause = ""
                    if isinstance(p_val, (int, float)):
                        sig_clause = (
                            f", a statistically significant difference (p = {p_val})" if p_val < 0.05
                            else f" (not statistically significant at p = {p_val})"
                        )
                    if isinstance(full_mean, (int, float)) and full_mean != 0:
                        pct = round((md / full_mean) * 100, 1)
                        interpretations.append(
                            f"- Under **{cfg}**, {label} {direction} by **{abs(pct)}%** relative to "
                            f"Full System (mean diff = {md}){sig_clause}, suggesting the evidence "
                            f"removed in this arm meaningfully {'contributes to' if md < 0 else 'detracts from'} "
                            f"{label.lower()} when present in the Full System."
                        )
                    else:
                        interpretations.append(
                            f"- Under **{cfg}**, {label} {direction} by a mean of **{md}** relative to "
                            f"Full System{sig_clause} (Full System baseline mean not available/zero, so a "
                            f"percentage change could not be computed)."
                        )
            if interpretations:
                for line in interpretations:
                    a(line)
            else:
                a("_Not Evaluated -- no paired metric had a real numeric mean difference to interpret._\n")
            if unchanged_by_config:
                a("")
                for cfg, labels in unchanged_by_config.items():
                    a(f"- Under **{cfg}**, no measured difference vs Full System: {', '.join(labels)}.")

    a("\n## 6e. Correlation Analysis\n")
    a(
        "Spearman rank correlation between the confidence level `confidence.assess_confidence()` "
        "actually assigned (mapped Low=1, Medium=2, High=3 -- the same ordering already encoded "
        "in that module's own rubric) and each quality metric, computed only over records with a "
        "real confidence level and a real numeric metric value. Reported `Not Evaluated` when "
        "fewer than 3 such paired observations exist, or when `scipy` is unavailable.\n"
    )
    if not correlations:
        a("_Correlation analysis was not computed for this run._\n")
    else:
        a("| Comparison | n | Spearman r | p-value |")
        a("|---|---|---|---|")
        for field_name, c in correlations.items():
            a(f"| {c.get('label', field_name)} | {c.get('n', 0)} | {c.get('coefficient', NOT_EVALUATED)} | "
              f"{c.get('p_value', NOT_EVALUATED)} |")

    a("\n## 6f. Error Analysis\n")
    a(
        "Computed strictly from the observed records above -- failure counts, parser fallback "
        "counts, response lengths, and confidence labels already captured on each `EvalRecord`.\n"
    )
    if not error_analysis:
        a("_Error analysis was not computed for this run._\n")
    else:
        ea_overall = error_analysis.get("overall", {})
        a(f"- Configuration with the highest failure rate: **{ea_overall.get('highest_failure_rate_config', NOT_EVALUATED)}** "
          f"(failure rate = {ea_overall.get('highest_failure_rate_value', NOT_EVALUATED)})")
        a(f"- Overall parser fallback frequency: **{ea_overall.get('parser_fallback_frequency', NOT_EVALUATED)}**")
        a(f"- Overall structured-response compliance rate: **{ea_overall.get('structured_response_compliance_rate', NOT_EVALUATED)}**")
        a(f"- Overall confidence distribution: {ea_overall.get('confidence_distribution', NOT_EVALUATED)}\n")

        a("| Configuration | n | Failures | Failure rate | Parser fallback freq. | Structured compliance rate | Avg. response length (chars) | Confidence distribution |")
        a("|---|---|---|---|---|---|---|---|")
        for cfg in ABLATION_CONFIGS:
            pc = error_analysis.get("per_config", {}).get(cfg, {})
            if "note" in pc:
                a(f"| {cfg} | 0 | - | - | - | - | - | - |  _{pc['note']}_")
                continue
            a(
                f"| {cfg} | {pc.get('n_records')} | {pc.get('n_failed')} | {pc.get('failure_rate')} | "
                f"{pc.get('parser_fallback_frequency', NOT_EVALUATED)} | "
                f"{pc.get('structured_response_compliance_rate', NOT_EVALUATED)} | "
                f"{pc.get('average_response_length_chars', NOT_EVALUATED)} | "
                f"{pc.get('confidence_distribution', NOT_EVALUATED)} |"
            )

    a("\n## 6g. Threats to Validity\n")
    a(
        "Standard IEEE-style disclosure of validity threats for this study. Each point below is "
        "either a structural property of the evaluation design (true regardless of this run's "
        "data) or is grounded in this run's own observed counts where noted.\n"
    )
    a("**Construct validity**")
    a(
        "- Heuristic metrics (Recommendation Quality, Evidence Fusion Correctness, Priority Focus "
        "Correctness -- \u00a76b.1b) are keyword/rule-based proxies for their underlying constructs, "
        "not semantic judgments; they can be gamed by keyword-stuffing and can under-score "
        "genuinely good paraphrased responses. They are not a substitute for human educator "
        "evaluation -- see the rubric template in \u00a76c, which this run does NOT auto-fill."
    )
    a(
        "- The Confidence label used in \u00a76e's correlation analysis is itself a rule-based "
        "evidence-completeness signal (see `confidence.py`), not an independent ground-truth "
        "quality judgment; correlating it with quality metrics measures internal consistency of "
        "the pipeline's own evidence bookkeeping, not external validity of that bookkeeping."
    )
    a("\n**Internal validity**")
    a(
        f"- The `Full System` configuration exposes fewer internal artifacts than the ablation "
        f"arms (`ask_mentor()`'s public return schema does not include retrieved_chunks, "
        f"shap_explanation, risk_reasons, or cognitive_state text -- \u00a76d), so several "
        f"metrics (Educational Usefulness, Grounding Recall and its sub-scores) are structurally "
        f"`Not Evaluated` for Full System and are only computable for the ablation arms, which "
        f"themselves call `build_student_context()` directly. This asymmetry means Full System "
        f"vs. ablation-arm comparisons for those specific metrics compare a real end-to-end call "
        f"against a partially-replayed one, not two calls with identical observability -- "
        f"disclosed here and in \u00a76d rather than papered over."
    )
    a(
        "- Ablation arms replay the same real functions `ask_mentor()` calls internally (\u00a72.2), "
        "but this replaying is itself a methodological necessity, not a literal `ask_mentor()` "
        "call; any behavior specific to `ask_mentor()`'s own internal control flow that this "
        "script does not replay would not be captured."
    )
    if error_analysis:
        ea_overall = error_analysis.get("overall", {})
        fb = ea_overall.get("parser_fallback_frequency", NOT_EVALUATED)
        if fb != NOT_EVALUATED:
            a(
                f"- {fb * 100:.1f}% of records with an observed parse mode in this run fell back to "
                f"the non-structured parse path (\u00a76f), which can inflate or deflate "
                f"text-derived metrics (e.g. Recommendation Quality, Evidence Fusion) relative to a "
                f"run with a higher structured-compliance rate."
            )
    a("\n**External validity**")
    a(
        f"- Results are based on the specific evaluated user population "
        f"({reproducibility.get('number_of_evaluated_users', NOT_EVALUATED) if reproducibility else NOT_EVALUATED} "
        f"user(s)) and question set "
        f"({reproducibility.get('number_of_evaluated_questions', NOT_EVALUATED) if reproducibility else NOT_EVALUATED} "
        f"question(s) -- \u00a71.3) in this run; they may not generalize to other students, personas, "
        f"risk profiles, or question phrasings not represented here."
    )
    a(
        "- The built-in default question set (when no `--questions-file`/`EVAL_QUESTIONS_FILE` is "
        "supplied) is a small, generic set of mentoring prompts and is not a validated benchmark; "
        "results using it should be read as a pilot-scale study, not a definitive benchmark result."
    )
    a("\n**Statistical conclusion validity**")
    a(
        "- Correlation coefficients in \u00a76e (and any relationship implied by paired comparisons "
        "in \u00a76b.4) describe association only; correlation does not imply causation, and no "
        "causal claim is made anywhere in this report beyond \"removing X was associated with a "
        "measured change in Y under this pipeline's specific implementation.\""
    )
    a(
        "- Sample sizes per (configuration, metric) cell are often small (see the `n` columns in "
        "\u00a76b.3 and \u00a76e); 95% CIs use a normal approximation and are noted as approximate for "
        "n < 30, and paired tests / correlations report `Not Evaluated` rather than a number when "
        "fewer than the required minimum of paired/valid observations exist, but even a computed "
        "p-value at small n should be read as indicative rather than conclusive."
    )
    a(
        "- Running many metric x configuration comparisons in \u00a76b.4 without a multiple-comparisons "
        "correction increases the chance that at least one comparison reaches p < 0.05 by chance; "
        "individual p-values there should be interpreted accordingly, not as a single confirmatory "
        "test."
    )

    a("\n## 6c. Human Evaluation Rubric (template -- not auto-filled)\n")
    a(
        "The metrics above are automated proxies. For an IEEE-style evaluation, pair them with "
        "human ratings on a real sample of (user, question, config) records. This is a rubric "
        "template only: this script does NOT invent human ratings. Populate it by having 2+ raters "
        "independently score a random sample (e.g. via `outputs/records.jsonl`), then report "
        "inter-rater agreement (e.g. Cohen's kappa) alongside the means.\n"
    )
    a("| Metric | Scale | Definition | Rater 1 (mean) | Rater 2 (mean) | Agreement |")
    a("|---|---|---|---|---|---|")
    for metric_name in [
        "Personalization", "Evidence Grounding", "Recommendation Quality",
        "Tone Appropriateness", "Actionability", "Overall Trust",
    ]:
        a(f"| {metric_name} | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |")

    a("\n## 6d. Limitations of the Response Quality Metrics (new, in addition to \u00a75)\n")
    a(
        "- These are automated proxies (keyword/token overlap), not semantic judgments; they can "
        "be gamed by keyword-stuffing and can under-score genuinely good paraphrased responses.\n"
        "- Evidence Fusion, Educational Usefulness, and all three Grounding sub-metrics require "
        "text that `ask_mentor()`'s public return schema does not expose (retrieved_chunks, "
        "shap_explanation, risk_reasons, cognitive_state content), so these read `Not Evaluated` "
        "for every `Full System` record in this run; only the ablation arms (which call "
        "`build_student_context()` directly) can be scored on those metrics.\n"
        "- Persona Consistency and Tone Differentiation need multiple observed responses per "
        "persona (and, for the latter, multiple personas) in the same run; small sample sizes "
        "will show as `Not Evaluated` or wide confidence intervals rather than a misleadingly "
        "precise number.\n"
        "- Paired comparisons only include pairs with real scores on both sides; configurations "
        "with many errored/`Not Evaluated` records will have few or zero valid pairs.\n"
    )

    a("\n## 7a. Reproducibility Summary\n")
    if not reproducibility:
        a("_Reproducibility summary was not computed for this run._\n")
    else:
        rp = reproducibility
        sha = rp.get("backend_sha256", NOT_EVALUATED)
        missing_mods = rp.get("backend_sha256_missing_modules")
        a("| Item | Value |")
        a("|---|---|")
        a(f"| Backend SHA256 (hash of imported backend module source) | `{sha}` |")
        if missing_mods:
            a(f"| Modules excluded from hash (not importable) | {', '.join(missing_mods)} |")
        a(f"| Ollama model | `{rp.get('ollama_model', NOT_EVALUATED)}` |")
        a(f"| Live LLM active | {rp.get('live_llm_active', NOT_EVALUATED)} |")
        a(f"| Number of evaluated users | {rp.get('number_of_evaluated_users', NOT_EVALUATED)} |")
        a(f"| Number of evaluated questions | {rp.get('number_of_evaluated_questions', NOT_EVALUATED)} |")
        a(f"| Number of ablation configurations | {rp.get('number_of_ablation_configs', NOT_EVALUATED)} |")
        a(f"| Total evaluations (users x questions x configs) | {rp.get('total_evaluations', NOT_EVALUATED)} |")
        a(f"| Timestamp (UTC) | {rp.get('timestamp_utc', NOT_EVALUATED)} |")

    a("\n## 7. Reproducing This Run\n")
    a(
        "```\n"
        "cd <backend project root>   # where the `services` package lives\n"
        "export EVAL_USER_IDS=\"<real_user_id_1>,<real_user_id_2>\"\n"
        "export USE_LIVE_LLM=true\n"
        "python ml/rag_ablation/rag_ablation.py\n"
        "```\n"
        "Every value in this report is derived from `outputs/records.jsonl`, produced by this "
        "exact run; re-running with the same user IDs, questions, backend code, and a live Ollama "
        "instance should reproduce the same real-data metrics up to LLM sampling variance "
        "(temperature=0.5, top_p=0.9, as configured in the real `mentor_service.ask_mentor`)."
    )

    return "\n".join(lines)


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-ids", type=str, default=None,
                         help="Comma-separated real evaluation user IDs.")
    parser.add_argument("--questions-file", type=str, default=None,
                         help="Path to newline-delimited real evaluation questions.")
    parser.add_argument("--backend-root", type=str, default=None,
                         help="Path to the folder containing the `services` package.")
    parser.add_argument("--no-live-llm", action="store_true",
                         help="Force offline/deterministic evaluation even if Ollama is reachable.")
    args = parser.parse_args()

    if args.backend_root:
        os.environ["BACKEND_ROOT"] = args.backend_root

    print("Importing real backend modules...")
    _try_import_backend()
    (OUTPUT_DIR / "backend_status.json").write_text(json.dumps(BACKEND_STATUS, indent=2), encoding="utf-8")
    for key, status in BACKEND_STATUS.items():
        print(f"  {key}: {'OK' if status['available'] else 'UNAVAILABLE -> ' + str(status['error'])}")

    use_live_env = os.environ.get("USE_LIVE_LLM", "true").lower() in ("1", "true", "yes")
    ollama_reachable, ollama_error = _check_ollama_reachable() if (use_live_env and not args.no_live_llm) else (False, None)
    live_llm_active = use_live_env and (not args.no_live_llm) and ollama_reachable
    print(f"Live LLM active: {live_llm_active} (USE_LIVE_LLM={use_live_env}, "
          f"--no-live-llm={args.no_live_llm}, ollama_reachable={ollama_reachable})")

    model_name = _infer_model_name()

    user_ids = resolve_eval_user_ids(args.user_ids)
    questions = resolve_eval_questions(args.questions_file)

    if not user_ids:
        print(
            "\nNo real evaluation user IDs were resolved (no --user-ids, no EVAL_USER_IDS env "
            "var, no rows in an `evaluation_users` Supabase table). Per the study's constraints, "
            "this script will NOT invent user IDs or synthetic students. Writing a report that "
            "states this explicitly and exiting.\n"
        )
        summary = aggregate([])
        quality_summary = aggregate_quality([], [])
        error_analysis = _error_analysis([])
        correlations = _correlation_analysis([], [])
        reproducibility = build_reproducibility_summary([], [], questions, model_name, live_llm_active)
        key_findings = build_key_findings(summary, quality_summary, error_analysis, correlations)
        figure_status = generate_figures(summary, quality_summary, _raw_quality_scores=[])
        report = generate_report(
            [], summary, figure_status, [], questions, live_llm_active, ollama_error, model_name,
            quality_summary=quality_summary,
            error_analysis=error_analysis,
            correlations=correlations,
            key_findings=key_findings,
            reproducibility=reproducibility,
        )
        (REPORT_DIR / "rag_ablation_report.md").write_text(report, encoding="utf-8")
        (OUTPUT_DIR / "records.jsonl").write_text("", encoding="utf-8")
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "quality_records.jsonl").write_text("", encoding="utf-8")
        (OUTPUT_DIR / "quality_summary.json").write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")
        # NEW output files (additive; existing files above are unchanged)
        (OUTPUT_DIR / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "key_findings.json").write_text(json.dumps(key_findings, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
        return 0

    print(f"Evaluating {len(user_ids)} user(s) x {len(questions)} question(s) x {len(ABLATION_CONFIGS)} configuration(s)...")

    records: List[EvalRecord] = []
    records_path = OUTPUT_DIR / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for user_id in user_ids:
            for question in questions:
                for cfg in ABLATION_CONFIGS:
                    print(f"  -> user={user_id!r} config={cfg!r} question={question[:40]!r}...")
                    rec = evaluate_one(user_id, question, cfg, live_llm_active, model_name)
                    records.append(rec)
                    f.write(json.dumps(asdict(rec)) + "\n")

    summary = aggregate(records)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- NEW: response-quality metrics (additive; does not alter records.jsonl
    # or summary.json above, which remain exactly as before) ---
    quality_scores = evaluate_response_quality(records)
    quality_summary = aggregate_quality(records, quality_scores)
    (OUTPUT_DIR / "quality_records.jsonl").write_text(
        "\n".join(json.dumps(asdict(q)) for q in quality_scores), encoding="utf-8"
    )
    (OUTPUT_DIR / "quality_summary.json").write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")

    # --- NEW: correlation analysis, error analysis, key findings, and
    # reproducibility summary (all additive; computed purely from the same
    # `records` / `quality_scores` already produced above). ---
    error_analysis = _error_analysis(records)
    correlations = _correlation_analysis(records, quality_scores)
    reproducibility = build_reproducibility_summary(records, user_ids, questions, model_name, live_llm_active)
    key_findings = build_key_findings(summary, quality_summary, error_analysis, correlations)
    (OUTPUT_DIR / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "key_findings.json").write_text(json.dumps(key_findings, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    figure_status = generate_figures(summary, quality_summary, _raw_quality_scores=quality_scores)

    report = generate_report(
        records, summary, figure_status, user_ids, questions, live_llm_active, ollama_error, model_name,
        quality_summary=quality_summary,
        error_analysis=error_analysis,
        correlations=correlations,
        key_findings=key_findings,
        reproducibility=reproducibility,
    )
    (REPORT_DIR / "rag_ablation_report.md").write_text(report, encoding="utf-8")

    print(f"\nDone. {len(records)} records written to {records_path}")
    print(f"Report: {REPORT_DIR / 'rag_ablation_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
