# 03 — High Risk Student Profile
## EduGuard AI Human Evaluation — Survey Case 3

> **Data provenance:** Produced by running the frozen `academic_risk_xgboost.pkl` on the held-out test partition (n=6,519, random_state=42). This student has predicted probability = 0.9998 — the **highest probability** of all 6,519 test students. The actual outcome was at-risk (class=1) — prediction is correct.

---

## SECTION A — TECHNICAL VERSION (Version A)

### Student Identification

| Field | Value |
|---|---|
| **Anonymized ID** | S-HR-003 |
| **Dataset** | OULAD held-out test partition |
| **Test Partition Index** | 3131 |
| **Correct Prediction** | ✅ Yes (predicted 1, actual label 1) |

---

### Risk Assessment

| Metric | Value |
|---|---|
| **Predicted Probability** | 0.9998 (99.98%) |
| **Risk Score (0–100)** | 99 / 100 |
| **Risk Level** | 🔴 **High** |
| **Predicted Class** | 1 (At risk) |
| **Actual Outcome** | 1 (At risk — Withdrawn/Failed) — **Prediction Correct** |

**Risk reasons computed by system:**
1. "High inactivity detected (32 days)"
2. "Student engagement trend declining"
3. "Academic performance weakening"

---

### Behavioural Features (raw values from model input)

| Feature | Value | Risk Direction |
|---|---|---|
| Total Interaction Clicks | 312 | 🚨 Very low |
| Mean Assessment Score | 1.0 | 🚨 Near-zero (critically low) |
| Active Learning Days | 24 | 🚨 Very low |
| Engagement Variability | 2.78 | Low (very little day-to-day variation — consistent non-engagement) |
| Inactivity Duration (days) | 32.0 | 🚨 Extended absence |
| Engagement Trend (Slope) | −0.304 | 🚨 Strongly declining |
| Assessment Consistency (SD) | 0.0 | ⚠️ SD=0 indicates only one or zero assessments submitted |

---

### SHAP Explanation (per-student feature attribution)

**SHAP Base Value:** 0.1206 (baseline risk for an average student)  
**Total SHAP Contribution:** +8.357 (overwhelmingly raises predicted risk from baseline)

#### Risk-Increasing Features (all 7 features push toward higher risk — no protective factors):

| Feature | Feature Value | SHAP Value | Effect |
|---|---|---|---|
| Inactivity Duration (days) | 32.0 | **+2.260** | Strongly increases risk (#1) |
| Assessment Consistency (SD) | 0.0 | **+1.924** | Strongly increases risk (#2) |
| Mean Assessment Score | 1.0 | **+1.721** | Strongly increases risk (#3) |
| Active Learning Days | 24.0 | **+1.379** | Increases risk (#4) |
| Engagement Trend (Slope) | −0.304 | **+0.685** | Increases risk (#5) |
| Total Interaction Clicks | 312.0 | **+0.315** | Increases risk (#6) |
| Engagement Variability | 2.78 | **+0.073** | Increases risk (#7) |

#### Risk-Decreasing Features (protective): **None**

**Key SHAP insight:** Every single feature in this student's profile pushes the model toward predicting at-risk. There are no counterbalancing protective factors. The three dominant drivers are:
1. Extended inactivity (32 days, SHAP +2.26)
2. Assessment consistency = 0.0 SD — this means the student has submitted so few assessments that there is no variation at all (likely only 0–1 assessment submitted)
3. Near-zero average score of 1.0 — critically low academic performance

> **Note on Assessment Consistency SD = 0.0:** A standard deviation of 0 means either exactly one assessment was submitted (SD undefined, stored as 0) or all submitted assessments received the same score (0 or near-0). Combined with avg_score = 1.0, this strongly suggests the student either did not submit assessments or received near-zero scores on all of them. The SHAP system correctly identifies this as a strong risk signal.

---

### Learner Archetype (Persona)

| Field | Value |
|---|---|
| **K-Means Cluster** | Cluster 1 |
| **Archetype Label** | **Burnout Pattern** |
| **Intervention Style** | Mental wellness intervention |
| **Source** | Frozen `persona_kmeans.pkl` + `advanced_persona_profiles.csv` |
| **Heuristic Agreement** | ✅ Both K-Means and heuristic agree: Burnout Pattern |

---

### Cognitive State

| Metric | Value | Interpretation |
|---|---|---|
| Attention Score | 77.57 | Moderate-high attention |
| Confusion Score | 22.43 | Low — below 70 threshold |
| Boredom Score | 10.0 | Low boredom |
| Cognitive Overload Score | 65.0 | Approaching threshold (below 70) |
| Stability Score | 69.93 | Moderate-high stability |

> *Source: Pre-computed in advanced_persona_profiles.csv.*

---

### Priority Focus (computed by determine_priority_focus())

| Field | Value |
|---|---|
| **Focus ID** | `burnout` |
| **Instruction to AI** | "Burnout risk is severe — prioritize workload reduction and recovery over anything else; do not assign new work." |
| **Reason** | persona=Burnout Pattern, risk_level=High |

> **This is the highest-priority branch in the system's cascade.** It fires when persona = "Burnout Pattern" AND risk_level = "High" simultaneously. It is treated as a near-safety-critical signal — the system is explicitly designed NOT to push more tasks at this student.

---

### Confidence Assessment

| Metric | Value |
|---|---|
| **Confidence Level** | **High** |
| **Has Risk Score** | ✅ Yes |
| **Has Persona** | ✅ Yes |
| **Has SHAP** | ✅ Yes |
| **Has Cognitive State** | ✅ Yes |
| **Needs Human Review** | ⚠️ Yes (offline mode; strongly recommended for any high-risk student regardless) |

**Confidence Rationale:** Risk score, SHAP-based feature explanation, and a known learner persona are all available, giving a quantified, explained, and contextualized basis for this response. Cognitive-state data was also available as a supporting signal.

---

### AI Mentor — Student Recommendation

> **[VERBATIM FROM SYSTEM — `_PERSONA_ACTIONS["Burnout Pattern"]["student"]`]**

*"Reduce your workload this week and schedule one real recovery break before your next study session."*

---

### AI Mentor — Educator Recommendation

> **[VERBATIM FROM SYSTEM — `_PERSONA_ACTIONS["Burnout Pattern"]["educator"]`]**

*"Flag for a workload/wellness check-in; consider a short extension or reduced load."*

---

### Psychological Framework

> **[VERBATIM FROM SYSTEM — `_PSYCHOLOGICAL_FRAMEWORKS["Burnout Pattern"]`]**

*"CBT — workload reduction and recovery: break tasks into tiny steps, challenge all-or-nothing thinking, explicitly recommend reducing load this week and building in a recovery break; flag for educator outreach rather than pushing more output."*

---

### Conversation Strategy

| Element | Value |
|---|---|
| **Opening Goal** | Reassure and validate; do not push more work |
| **Teaching Style** | Brief, low-pressure, permission-giving tone |
| **Recommendation Style** | One recovery-oriented micro-action |
| **Educator Goal** | Flag for a workload/wellness check-in |

---

### Supporting Figures

| Figure | File | Purpose |
|---|---|---|
| **SHAP Waterfall (generated)** | `../ieee_results/figures/eval_waterfall_high_risk.png` | Per-student feature attribution for S-HR-003 |
| SHAP Global Importance | `../ieee_results/figures/fig3_shap_importance.png` | Context: global feature rankings |
| SHAP Beeswarm | `../ieee_results/figures/fig4_shap_beeswarm.png` | Distribution across all students |
| Persona Clusters | `../ieee_results/figures/fig_g2_tsne_clusters.png` | Cluster 1 = Burnout Pattern |

---

## SECTION B — EDUCATOR VERSION (Version B)

> *Technical explanations rewritten into plain language. AI Mentor recommendations are unchanged — verbatim as produced by the system.*

---

### About This Student

**Student ID:** S-HR-003  
**Risk Level:** 🔴 High Risk  
**AI System Confidence:** High — All required information was available. **This was the highest-risk prediction in the entire test set of 6,519 students.** The actual outcome confirmed the student had withdrawn or failed.

---

### What the AI System Found

This student shows the most concerning pattern of all students in the dataset. Nearly every signal the system measured is pointing in the wrong direction.

**The core concerns:**
- **Average assessment score of 1.0 out of 100** — essentially no academic activity recorded. Either very few assessments were submitted, or those submitted scored near-zero.
- **Only 24 active days** — the student was present on a very small number of days.
- **32 days of consecutive inactivity** — the student had been absent for over a month.
- **Engagement declining sharply** — the downward trend is among the steepest in the dataset.
- **Only 312 total interactions** — a very low level of activity across the entire module.

**What makes this case especially clear:**
There are **no protective factors**. Every single measure the AI system looks at is pointing toward risk. The system's explanation (SHAP) confirms this: all 7 features push the prediction toward at-risk. This is rare — most students have at least some protective signals.

---

### Why This Risk Level?

The three most important factors (in order of their influence on the prediction):

1. **32 days of inactivity** — the single largest driver. Extended absence is the #1 predictor of disengagement and failure in this system.
2. **Assessment score variation = 0** — this means the student either submitted nothing or submitted so few pieces of work that there is nothing to measure. This is a very strong signal of disengagement.
3. **Near-zero average score (1.0/100)** — the assessments that were submitted received extremely low scores.

**Learner Type:** This student is classified as a **Burnout Pattern** — the archetype associated with overwhelm, exhaustion, or a situation where the student appears to have stopped engaging entirely. Both the K-Means clustering model and the heuristic reasoning agreed on this classification.

---

### AI Mentor Recommendation — For the Student

> **Produced by the AI Mentor system. Not paraphrased or edited.**

*"Reduce your workload this week and schedule one real recovery break before your next study session."*

---

### AI Mentor Recommendation — For the Educator

> **Produced by the AI Mentor system. Not paraphrased or edited.**

*"Flag for a workload/wellness check-in; consider a short extension or reduced load."*

---

### Why This Recommendation?

This is the most important design decision in EduGuard AI: **the system recommends workload reduction, not "study harder."**

The AI system is designed to recognize that when a student has stopped engaging at this level, pushing more content or assessment deadlines at them is likely to backfire. The psychological approach is CBT-based: break tasks into the smallest possible steps, validate that the student is struggling, and explicitly give permission to reduce effort before rebuilding.

The educator recommendation calls for a **wellness check-in** rather than a performance warning. This is intentional — the system treats the student's disengagement as potentially driven by personal or emotional circumstances, not merely a study effort problem.

**Note:** The system's recommendation to "not assign new work" is a hard instruction to the AI Mentor — it is one of only two explicit safety guardrails in the system (the other is "do not give remedial advice to Consistent Learners"). The system will not violate this instruction.

---

### Confidence and Human Review

The AI system rated its confidence as **High** — all evidence was available and consistent. However, **immediate human educator intervention is strongly recommended**. The AI system is designed to support human decision-making, not replace it. For a student at this risk level, a personal outreach from a real educator is the most important next step.
