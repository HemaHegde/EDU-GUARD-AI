# EduGuard AI Mentor V2 — Sprint 1

## Files

- `context_builder.py` — gathers profile, risk, persona, SHAP (optional),
  cognitive state (optional), behaviour summary, and chat history into a
  single `StudentContext`. Never fabricates missing optional data.
- `prompt_builder.py` — turns a `StudentContext` + retrieved chunks into
  a structured, section-labeled system prompt.
- `retrieval.py` — isolates the FAISS index + embedding model + lookup,
  unchanged in behaviour from V1.
- `confidence.py` — computes High/Medium/Low confidence purely from
  which evidence was actually available (risk score, persona, SHAP,
  risk reasons, cognitive state, retrieved material). The LLM is never
  asked to self-report confidence.
- `mentor_service.py` — orchestrator. Public entry point `ask_mentor(user_id, question)`
  is unchanged in signature and keeps all original return keys; new
  keys are additive only.

## Backward compatibility

- Same function name and signature: `ask_mentor(user_id: str, question: str)`.
- Original return keys (`status`, `question`, `mentor_response`, `persona`,
  `risk_score`, `risk_level`) are all still present and mean the same
  thing as before.
- `mentor_history` table is written with the exact same columns as V1
  (`user_id`, `question`, `response`, `created_at`) — no schema change.
- No FastAPI route changes required. If your route currently does
  `from services.mentor_service import ask_mentor`, it keeps working
  with no code change on the route side; new fields in the response
  dict are simply available if/when the route or frontend wants them.

## New fields in the response

| Field | Type | Notes |
|---|---|---|
| `student_recommendation` | str or None | Grounded in risk+SHAP+persona+retrieved material. States "not enough data" rather than inventing advice when evidence is thin. |
| `educator_recommendation` | str or None | Same grounding, framed for the instructor. |
| `confidence` | "High"/"Medium"/"Low" or None | Computed from evidence availability only — see `confidence.py` rubric. |
| `confidence_rationale` | str or None | Plain-language explanation of why that confidence level was assigned. |
| `evidence_used` | list[str] | Which signals actually fed into this response. |
| `shap_available` | bool | True only if a real SHAP explanation was retrieved. |
| `cognitive_state_available` | bool | True only if a real cognitive-state value was retrieved. |

## Integration points you still need to wire up

This sprint deliberately does **not** fabricate two upstream services
that the spec mentions as optional:

1. **`services/shap_service.py`** — expected to expose
   `get_shap_explanation(user_id) -> dict | None`. If this module
   doesn't exist yet in your project, `context_builder.py` catches the
   ImportError and proceeds with `shap_explanation = None`,
   `shap_available = False`. The prompt explicitly tells the LLM not to
   invent SHAP-driven specifics in that case.
2. **`services/cognitive_state_service.py`** — expected to expose
   `get_cognitive_state(user_id) -> dict | None`. Same fallback
   behavior if absent.

Until those services exist, the mentor will run at **Medium** or
**Low** confidence (per the rubric in `confidence.py`), which is the
correct, honest behavior rather than overclaiming.

## Not in scope for this sprint

- No changes to `risk_service.py` or `persona_service.py` internals —
  they're consumed as-is.
- No database schema changes.
- No new API routes — only the internals of the existing mentor call.
- Two-LLM-call patterns (e.g., a separate call purely for SHAP
  summarization) were intentionally avoided to keep latency and cost
  the same as V1; the structured three-section output is parsed from a
  single LLM call. If recommendation quality needs improvement later,
  splitting into dedicated calls per section is a natural Sprint 2 item.
