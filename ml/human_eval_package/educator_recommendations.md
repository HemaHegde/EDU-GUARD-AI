# Educator Recommendations — IEEE Access Human Evaluation
## EduGuard AI: AI Mentor Evaluation Study

> **Note for evaluators**: These recommendations are produced by the EduGuard AI backend
> recommendation engine and are shown here verbatim — not paraphrased or edited.
> They appear **separately** below each student screenshot in the Google Form.
> They are **NOT** visible to students.

---

## Scenario 1 — Consistent Learner (Low Risk)

**Student Profile:** S-LR-001  
**Risk Level:** Low  
**Learner Archetype:** Consistent Learner  
**Student Question:** "I've been doing well in my studies recently. What should I focus on next to continue improving?"

### Educator Recommendation

> Produced by the AI Mentor system. Not paraphrased or edited.
> Source: _PERSONA_ACTIONS["Consistent Learner"]["educator"]

"Consider this student for a peer-mentoring or leadership opportunity."

---

## Scenario 2 — Last-Minute Survivor (Moderate Risk)

**Student Profile:** S-MR-002  
**Risk Level:** Moderate  
**Learner Archetype:** Last-Minute Survivor  
**Student Question:** "I've been falling behind lately and often leave my coursework until the last minute. What is one practical step I can take to improve?"

### Educator Recommendation

> Produced by the AI Mentor system. Not paraphrased or edited.
> Source: _PERSONA_ACTIONS["Last-Minute Survivor"]["educator"]

"Consider a scaffolded reminder or interim checkpoint ahead of the next deadline."

---

## Scenario 3 — Burnout Pattern (High Risk)

**Student Profile:** S-HR-003  
**Risk Level:** High  
**Learner Archetype:** Burnout Pattern  
**Student Question:** "I've been feeling overwhelmed and haven't been studying for a while. What's one small step I can take today to get back on track?"

### Educator Recommendation

> Produced by the AI Mentor system. Not paraphrased or edited.
> Source: _PERSONA_ACTIONS["Burnout Pattern"]["educator"]

"Flag for a workload/wellness check-in; consider a short extension or reduced load."

---

## Source Provenance

All educator recommendations are verbatim outputs from the system's frozen recommendation
engine (recommendation_engine.py -> _PERSONA_ACTIONS mapping), documented in:

- 01_Low_Risk.md - Section: "AI Mentor - Educator Recommendation"
- 02_Moderate_Risk.md - Section: "AI Mentor - Educator Recommendation"
- 03_High_Risk.md - Section: "AI Mentor - Educator Recommendation"

These outputs are deterministic, persona-driven, hardcoded actions that do not change
between runs.
