# EduGuard-AI Ethics & Fairness Summary

## 1. Privacy Protection
Implemented at the prompt and backend API level:
- Student-facing outputs (`<MENTOR_REPLY>`, `<STUDENT_RECOMMENDATION>`) strictly prohibit exposing internal model confidence, SHAP explanations, or raw prediction probabilities.
- Educator-facing outputs (`<EDUCATOR_RECOMMENDATION>`) may safely include these analytical details for institutional review.

## 2. Misuse Prevention
Implemented in the `prompt_builder.py` instructions and `mentor_service.py` API:
- All generated recommendations clearly state they are advisory and not autonomous decisions.
- `needs_human_review` is automatically enabled whenever evidence quality is low, confidence is below 0.50, or a contradiction is detected in retrieved chunks.
- The LLM is explicitly instructed to refuse unsupported recommendations when evidence is insufficient.

## 3. Over-monitoring Analysis
- **Intervention Rate:** 46.62% (3039/6519 students flagged)
- **False Positive Rate:** 5.20% (Only 160 unnecessary interventions)
- The system demonstrates a strong bias against over-surveillance, with false positives significantly lower than false negatives.

See `over_monitoring_report.md` and the existing `fairness_report.md` for full demographic and statistical breakdowns.
