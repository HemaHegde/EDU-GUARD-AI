# 01 — Low Risk Student Profile
## EduGuard AI Human Evaluation — Survey Case 1

> **Data provenance:** Produced by running the frozen `academic_risk_xgboost.pkl` on the held-out test partition (n=6,519, random_state=42). This student is the **lowest-probability** case in the entire test set (argmin of predicted probabilities).

---

## SECTION A — TECHNICAL VERSION (Version A)

### Student Identification

| Field | Value |
|---|---|
| **Anonymized ID** | S-LR-001 |
| **Dataset** | OULAD held-out test partition |
| **Test Partition Index** | 1416 |
| **Correct Prediction** | ✅ Yes (predicted 0, actual label 0) |

---

### Risk Assessment

| Metric | Value |
|---|---|
| **Predicted Probability** | 0.0130 (1.30%) |
| **Risk Score (0–100)** | 1 / 100 |
| **Risk Level** | 🟢 **Low** |
| **Predicted Class** | 0 (Not at risk) |
| **Actual Outcome** | 0 (Not at risk) — Prediction Correct |

**Risk reasons computed by system:**
- "High inactivity detected (60 days)"

> *Note: Despite 60 days of inactivity, the other features (active days = 220, avg_score = 93.4) are so strongly protective that the model produces an extremely low risk score (1/100). The inactivity threshold condition fires, but is overwhelmed by the positive signals.*

---

### Behavioural Features (raw values from model input)

| Feature | Value | Risk Direction |
|---|---|---|
| Total Interaction Clicks | 7,058 | Protective (high activity) |
| Mean Assessment Score | 93.43 | Protective (excellent performance) |
| Active Learning Days | 220 | Protective (very high) |
| Engagement Variability | 6.66 | Neutral |
| Inactivity Duration (days) | 60.0 | ⚠️ Warning (>15 days) |
| Engagement Trend (Slope) | +0.075 | Protective (improving) |
| Assessment Consistency (SD) | 6.34 | Protective (very consistent) |

---

### SHAP Explanation (per-student feature attribution)

**SHAP Base Value:** 0.1206 (baseline risk for an average student)  
**Total SHAP Contribution:** −4.454 (strongly lowers predicted risk from baseline)

#### Risk-Increasing Features (push toward higher risk):

| Feature | Feature Value | SHAP Value | Effect |
|---|---|---|---|
| Inactivity Duration (days) | 60.0 | **+1.125** | Increases risk |

#### Risk-Decreasing Features (protective — push toward lower risk):

| Feature | Feature Value | SHAP Value | Effect |
|---|---|---|---|
| Active Learning Days | 220.0 | **−3.051** | Strongly decreases risk |
| Mean Assessment Score | 93.43 | **−1.238** | Decreases risk |
| Assessment Consistency (SD) | 6.34 | **−0.597** | Decreases risk |
| Total Interaction Clicks | 7,058 | −0.478 | Decreases risk |
| Engagement Trend (Slope) | +0.075 | −0.197 | Decreases risk |
| Engagement Variability | 6.66 | −0.017 | Decreases risk |

**Key SHAP insight:** The student's 220 active days alone contributes a SHAP value of −3.051, which is nearly 3× larger in magnitude than the inactivity warning (+1.125). The model correctly determines this student is low-risk despite the inactivity flag.

---

### Learner Archetype (Persona)

| Field | Value |
|---|---|
| **K-Means Cluster** | Cluster 4 |
| **Archetype Label** | **Consistent Learner** |
| **Intervention Style** | Advanced learning opportunities |
| **Source** | Frozen `persona_kmeans.pkl` + `advanced_persona_profiles.csv` |

---

### Cognitive State

| Metric | Value | Interpretation |
|---|---|---|
| Attention Score | 46.87 | Moderate attention |
| Confusion Score | 53.13 | Moderate — below 70 threshold (no confusion alert) |
| Boredom Score | 26.0 | Low boredom |
| Cognitive Overload Score | 73.0 | ⚠️ Elevated (above 70 — notable) |
| Stability Score | 55.74 | Moderate behavioural stability |

> *Source: Pre-computed in advanced_persona_profiles.csv from EdNet video intelligence pipeline (GRU-based cognitive model, ROC-AUC=0.878).*

---

### Priority Focus (computed by determine_priority_focus())

| Field | Value |
|---|---|
| **Focus ID** | `enrichment` |
| **Instruction to AI** | "This learner is stable and not at elevated risk — lead with enrichment or a leadership opportunity. Do not use remedial or risk-reduction framing anywhere in your reply." |
| **Reason** | persona=Consistent Learner, risk_level=Low |

---

### Confidence Assessment

| Metric | Value |
|---|---|
| **Confidence Level** | **High** |
| **Has Risk Score** | ✅ Yes |
| **Has Persona** | ✅ Yes |
| **Has SHAP** | ✅ Yes |
| **Has Cognitive State** | ✅ Yes |
| **Needs Human Review** | ⚠️ Yes (offline mode: retrieval pipeline unavailable; in live API this would be resolved by retrieval similarity scoring) |

**Confidence Rationale:** Risk score, SHAP-based feature explanation, and a known learner persona are all available, giving a quantified, explained, and contextualized basis for this response. Cognitive-state data was also available as a supporting signal.

---

### AI Mentor — Student Recommendation

> **[VERBATIM FROM SYSTEM — `_PERSONA_ACTIONS["Consistent Learner"]["student"]`]**

*"Take on an enrichment activity or advanced topic — current performance supports it."*

---

### AI Mentor — Educator Recommendation

> **[VERBATIM FROM SYSTEM — `_PERSONA_ACTIONS["Consistent Learner"]["educator"]`]**

*"Consider this student for a peer-mentoring or leadership opportunity."*

---

### Psychological Framework

> **[VERBATIM FROM SYSTEM — `_PSYCHOLOGICAL_FRAMEWORKS["Consistent Learner"]`]**

*"SDT — enrichment and growth: affirm what is already working, then offer a stretch opportunity — advanced material, a peer-mentoring or leadership role — rather than remedial advice; this student does not need risk-reduction framing."*

---

### Conversation Strategy

| Element | Value |
|---|---|
| **Opening Goal** | Affirm what's already working |
| **Teaching Style** | Forward-looking and growth-oriented; no risk or remedial framing |
| **Recommendation Style** | One stretch/enrichment opportunity |
| **Educator Goal** | Consider for a peer-mentoring or leadership role |

---

### Supporting Figures

| Figure | File | Purpose |
|---|---|---|
| **SHAP Waterfall (generated)** | `../ieee_results/figures/eval_waterfall_low_risk.png` | Per-student feature attribution for S-LR-001 |
| SHAP Global Importance | `../ieee_results/figures/fig3_shap_importance.png` | Context: global feature rankings |
| SHAP Beeswarm | `../ieee_results/figures/fig4_shap_beeswarm.png` | Distribution across all students |
| Persona Clusters | `../ieee_results/figures/fig_g2_tsne_clusters.png` | Cluster 4 = Consistent Learner |

---

## SECTION B — EDUCATOR VERSION (Version B)

> *In this version, the technical explanations have been rewritten into plain language. The AI Mentor recommendations are unchanged — they appear verbatim exactly as produced by the system.*

---

### About This Student

**Student ID:** S-LR-001  
**Risk Level:** 🟢 Low Risk  
**AI System Confidence:** High — All required information was available to make this assessment.

---

### What the AI System Found

The EduGuard AI system reviewed seven aspects of how this student engages with their online course.

**What this student is doing well:**
- They logged into the course platform on **220 separate days** — this is very high activity compared to other students.
- Their average assessment score is **93.4 out of 100** — among the strongest in the dataset.
- Their performance is **very consistent** — their scores don't fluctuate much from one assignment to the next.
- Their engagement is **improving over time** — they're becoming more active, not less.
- They have accumulated over **7,000 interaction clicks** with course materials.

**One potential concern (but not a red flag here):**
- The student has been absent for **60 days** at some point during the course. Normally this would be a warning. However, the AI's explanation system (SHAP) determined that the student's high activity on other days strongly outweighs this absence — it accounts for about 1/3 of the student's overall low-risk rating compared to the active days which account for over 3× that protective effect.

---

### Why This Risk Level?

The AI system's feature explanation (SHAP) shows:
- **Biggest reason for low risk:** The student logged in on 220 days — this single factor contributes most strongly to the low-risk prediction.
- **Second biggest reason:** High average assessment scores (93.4).
- **Only concern:** 60 days of inactivity — but this is outweighed 3:1 by the protective factors.

**Learner Type:** This student is classified as a **Consistent Learner** — someone who engages regularly and performs reliably.

---

### AI Mentor Recommendation — For the Student

> **Produced by the AI Mentor system. Not paraphrased or edited.**

*"Take on an enrichment activity or advanced topic — current performance supports it."*

---

### AI Mentor Recommendation — For the Educator

> **Produced by the AI Mentor system. Not paraphrased or edited.**

*"Consider this student for a peer-mentoring or leadership opportunity."*

---

### Why This Recommendation?

The AI system is designed to **not give remedial advice to students who are not at risk**. This is a deliberate design choice: students at low risk who are performing well do not need "study harder" messages — they need growth opportunities. The system therefore recommends an enrichment path or a leadership/mentoring role.

---

### Confidence and Human Review

The AI system rated its confidence in this assessment as **High** — it had all the information it needed: a risk score, an explanation of which factors drove the assessment, and a known learner profile.

The system also notes that **a human educator should review this case before taking any action** (as it does for all cases when running in offline mode). In practice, for a clearly low-risk student like this, the review is a brief confirmation rather than a complex judgment.
