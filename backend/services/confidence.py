"""
confidence.py

Derives a confidence label (High / Medium / Low) for a mentor response,
based ONLY on which pieces of evidence were actually available when the
response was generated.

This is intentionally simple and auditable: confidence is a function of
evidence completeness, not of how fluent the LLM's answer sounds. We
never ask the LLM to self-report confidence, since that would be the
model fabricating a number with no grounding.
"""

from dataclasses import dataclass
from typing import List

from .context_builder import StudentContext


@dataclass
class ConfidenceResult:
    level: str  # "High" | "Medium" | "Low"
    rationale: str
    evidence_used: List[str]


def assess_confidence(
    context: StudentContext,
    retrieved_chunks_found: bool,
) -> ConfidenceResult:
    """
    Confidence rubric:

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

    return ConfidenceResult(
        level=level,
        rationale=rationale,
        evidence_used=evidence_used,
    )
