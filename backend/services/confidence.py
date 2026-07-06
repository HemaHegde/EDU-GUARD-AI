"""
confidence.py

Derives a confidence assessment for a mentor response, based ONLY on
which pieces of evidence were actually available when the response was
generated. This is intentionally simple and auditable: confidence is a
function of evidence completeness/quality, not of how fluent the LLM's
answer sounds. We never ask the LLM to self-report confidence, since
that would be the model fabricating a number with no grounding.

SPRINT 8 UPDATE (HALLUCINATION SAFETY HARDENING):
The original High/Medium/Low rubric (`level`, `rationale`,
`evidence_used`) is UNCHANGED — same three branches, same evidence
fields, same wording. This sprint adds a second, complementary
assessment focused specifically on RETRIEVAL evidence quality, since
the existing rubric only looked at risk/persona/SHAP and had no notion
of "was the retrieved learning material any good":

  - `normalized_score`   : a single 0..1 confidence number combining
                            retrieval similarity, how many chunks
                            support the answer, and how consistent
                            those chunks are with each other (Req:
                            Confidence Estimation).
  - `max_similarity`      : the strongest similarity score seen among
                            the retrieved chunks (or None if nothing
                            was retrieved), passed straight through
                            from retrieval.py's already-computed
                            per-chunk `similarity_score` — not
                            recomputed independently.
  - `supporting_chunk_count`: how many retrieved chunks actually carry
                            a similarity score (i.e. how much evidence
                            volume backs the answer).
  - `retrieval_consistency`: 0..1 agreement measure between the
                            retrieved chunks (see
                            `_compute_retrieval_consistency`).
  - `contradiction_detected`: True when retrieved chunks disagree
                            enough (low consistency) that they may be
                            answering from unrelated/contradictory
                            material (Req: Educator Review Flag).
  - `needs_human_review`  : True whenever normalized_score or
                            max_similarity falls below their
                            respective configurable thresholds, or
                            contradiction_detected is True, or the
                            original High/Medium/Low rubric already
                            landed on "Low" (Req: Educator Review
                            Flag).

All of this is computed from data ALREADY available on `StudentContext`
and the `retrieved_chunks` list retrieval.py already returns — no new
upstream call, no new stored value, no new fabricated fact. Consistent
with this file's original design goal, none of it asks the LLM to
self-report anything.

CLARIFICATION (additive-only, no logic change): `level` and
`normalized_score` were being surfaced to API consumers as "confidence"
and "confidence_score", which reads as two contradictory measurements
of the same thing (e.g. "High" alongside "0.39"). They are NOT the
same thing:

  - `level` (High/Medium/Low)   -> STUDENT_RISK_CONFIDENCE. Confidence
     in the underlying psychological/behavioural risk assessment (risk
     score + SHAP + persona availability). This is the ORIGINAL,
     UNCHANGED rubric from this file's first version.

  - `normalized_score` (0..1)   -> MENTOR_RESPONSE_CONFIDENCE. Confidence
     that the RETRIEVED learning material is strong enough to ground
     the LLM's mentor response (similarity + chunk count + cross-chunk
     consistency). This is the Sprint 8 addition.

A learner can simultaneously have a well-understood risk profile (High
student-risk confidence) and a question with weak retrieval hits (Low
mentor-response confidence) — that is not a contradiction, it is two
different signals about two different parts of the pipeline. Two
read-only aliases are added below purely for naming clarity:
`student_risk_confidence` (== `level`) and `mentor_response_confidence`
(== `normalized_score`). No existing field was renamed or removed, so
no existing caller of `.level` or `.normalized_score` breaks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_builder import StudentContext
from .retrieval import SIMILARITY_THRESHOLD, TOP_K


# =========================================================
# CONFIGURABLE PARAMETERS (Hallucination Safety Hardening)
# =========================================================
# Minimum acceptable normalized confidence score (0..1) before a
# response is flagged for educator review. This module owns confidence
# concerns, so it owns this threshold; retrieval.py owns
# SIMILARITY_THRESHOLD (imported above) since that one is a property of
# retrieval quality specifically. Both are starting heuristic values —
# tune empirically for this deployment.
CONFIDENCE_THRESHOLD = 0.5

# Below this fraction of retrieved chunks agreeing on the same
# topic/source, we treat the evidence set as internally inconsistent
# (possibly contradictory) rather than just "thin". See
# `_compute_retrieval_consistency`.
_CONSISTENCY_CONTRADICTION_FLOOR = 0.5

# Weights for combining the three retrieval-quality signals into one
# normalized_score. Similarity is weighted highest because it is the
# most direct measure of "does this evidence actually match the
# question"; consistency is weighted next because contradictory
# evidence is specifically dangerous for hallucination risk (citing
# two disagreeing sources is worse than citing one good source);
# chunk count is weighted lowest because one highly relevant chunk is
# worth more than several mediocre ones. Weights sum to 1.0.
_WEIGHT_SIMILARITY = 0.5
_WEIGHT_CONSISTENCY = 0.3
_WEIGHT_CHUNK_COUNT = 0.2


@dataclass
class ConfidenceResult:
    level: str  # "High" | "Medium" | "Low"
    rationale: str
    evidence_used: List[str]

    # Hallucination Safety Hardening additions (all additive fields):
    normalized_score: float = 0.0
    max_similarity: Optional[float] = None
    supporting_chunk_count: int = 0
    retrieval_consistency: float = 0.0
    contradiction_detected: bool = False
    needs_human_review: bool = False

    # -------------------------------------------------------------
    # Naming-clarity aliases (Issue 1 fix, additive-only).
    # These do NOT introduce new data or new computation — they just
    # give the two distinct confidence concepts distinct, self-
    # documenting names so API consumers (and this file's own
    # docstring) never have to say "confidence" when they mean two
    # different things. `level`/`normalized_score` are kept as the
    # canonical fields so no existing internal caller breaks.
    # -------------------------------------------------------------
    @property
    def student_risk_confidence(self) -> str:
        """High/Medium/Low confidence in the psychological/behavioural
        risk assessment. Identical value to `level`; alias only."""
        return self.level

    @property
    def mentor_response_confidence(self) -> float:
        """0..1 confidence that retrieved evidence grounds the mentor's
        response. Identical value to `normalized_score`; alias only."""
        return self.normalized_score


def _compute_retrieval_consistency(retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Heuristic, auditable consistency measure between retrieved chunks:
    the fraction of chunks that share the same `source` (falling back
    to `topic`) as the majority label. This deliberately reuses
    metadata retrieval.py already attaches to every chunk rather than
    introducing any new model, embedding comparison, or NLP step —
    consistent with the "no new ML models" constraint on this project.

    1.0 = every retrieved chunk agrees on source/topic (highly
          consistent evidence, low risk of contradictory citations).
    Low value = chunks are drawn from unrelated sources/topics, which
          is treated as a signal the retrieved evidence may be
          contradictory or off-topic for this specific question.
    """
    if not retrieved_chunks:
        return 0.0

    labels = [
        chunk.get("source") or chunk.get("topic") or "unknown"
        for chunk in retrieved_chunks
    ]
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    majority_count = max(counts.values())
    return majority_count / len(labels)


def assess_confidence(
    context: StudentContext,
    retrieved_chunks_found: bool,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ConfidenceResult:
    """
    Confidence rubric (UNCHANGED from the original design):

    HIGH   - risk score is present AND SHAP explanation is present
             AND persona is known (not "Unknown Persona")
             -> we have a quantified risk, a feature-level explanation
                of WHY, and a behavioural archetype to contextualize it.

    MEDIUM - risk score is present AND persona is known, but SHAP is
             missing -> we know risk and behavioural pattern, but not
             the specific drivers behind the risk score.

    LOW    - risk score is missing, OR persona is unknown, OR there are
             no risk_reasons and no retrieved learning material to
             ground a recommendation in -> we are working with thin
             evidence and the response should say so rather than
             confidently inventing advice.

    Cognitive state is treated as a bonus signal (noted in rationale)
    but does not by itself upgrade a Low/Medium to a higher tier, since
    it's an optional/experimental signal in this system.

    SPRINT 8 ADDITION — RETRIEVAL-EVIDENCE CONFIDENCE (Req: Confidence
    Estimation): independently of the level/rationale logic above, this
    also computes a normalized_score in [0, 1] from three retrieval-
    specific signals (max similarity, supporting chunk count,
    cross-chunk consistency), and a needs_human_review flag (Req:
    Educator Review Flag). `retrieved_chunks` is OPTIONAL and defaults
    to None so existing callers that only pass `retrieved_chunks_found`
    keep working exactly as before — they simply get conservative
    defaults (normalized_score=0.0 from missing similarity data,
    needs_human_review driven only by the level/similarity checks that
    remain meaningful without per-chunk data).
    """

    has_risk_score = context.risk_score is not None
    has_persona = context.persona not in (None, "", "Unknown Persona")
    has_shap = context.shap_available and context.shap_explanation is not None
    has_reasons = len(context.risk_reasons) > 0
    has_cognitive_state = (
        context.cognitive_state_available
        and context.cognitive_state is not None
    )

    evidence_used = []
    if has_risk_score:
        evidence_used.append("risk_score")
    if has_persona:
        evidence_used.append("persona")
    if has_shap:
        evidence_used.append("shap_explanation")
    if has_reasons:
        evidence_used.append("risk_reasons")
    if has_cognitive_state:
        evidence_used.append("cognitive_state")
    if retrieved_chunks_found:
        evidence_used.append("retrieved_learning_material")

    if has_risk_score and has_persona and has_shap:
        level = "High"
        rationale = (
            "Risk score, SHAP-based feature explanation, and a known "
            "learner persona are all available, giving a quantified, "
            "explained, and contextualized basis for this response."
        )
    elif has_risk_score and has_persona:
        level = "Medium"
        rationale = (
            "Risk score and learner persona are available, but no "
            "SHAP explanation was available to confirm the specific "
            "drivers behind the risk score."
        )
    else:
        level = "Low"
        missing = []
        if not has_risk_score:
            missing.append("risk score")
        if not has_persona:
            missing.append("learner persona")
        rationale = (
            "Limited evidence available"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + ". Response should be treated as general guidance rather "
            "than a personalized, evidence-backed recommendation."
        )

    if has_cognitive_state:
        rationale += " Cognitive-state data was also available as a supporting signal."

    # -------------------------
    # Retrieval-evidence confidence (Sprint 8 addition)
    # -------------------------
    chunks = retrieved_chunks or []
    similarities = [
        chunk.get("similarity_score")
        for chunk in chunks
        if isinstance(chunk.get("similarity_score"), (int, float))
    ]

    max_similarity = max(similarities) if similarities else None
    supporting_chunk_count = len(similarities)
    retrieval_consistency = _compute_retrieval_consistency(chunks)
    contradiction_detected = bool(chunks) and retrieval_consistency < _CONSISTENCY_CONTRADICTION_FLOOR

    similarity_component = max_similarity if max_similarity is not None else 0.0
    chunk_count_component = min(supporting_chunk_count / TOP_K, 1.0) if TOP_K else 0.0
    consistency_component = retrieval_consistency

    normalized_score = (
        _WEIGHT_SIMILARITY * similarity_component
        + _WEIGHT_CONSISTENCY * consistency_component
        + _WEIGHT_CHUNK_COUNT * chunk_count_component
    )
    normalized_score = round(max(0.0, min(1.0, normalized_score)), 2)

    needs_human_review = (
        normalized_score < confidence_threshold
        or (max_similarity is not None and max_similarity < similarity_threshold)
        or contradiction_detected
        or level == "Low"
    )

    return ConfidenceResult(
        level=level,
        rationale=rationale,
        evidence_used=evidence_used,
        normalized_score=normalized_score,
        max_similarity=max_similarity,
        supporting_chunk_count=supporting_chunk_count,
        retrieval_consistency=round(retrieval_consistency, 2),
        contradiction_detected=contradiction_detected,
        needs_human_review=needs_human_review,
    )
