# 02 — Moderate Risk Student Profile
## EduGuard AI Human Evaluation — Survey Case 2

> **Data provenance:** Produced by running the frozen `academic_risk_xgboost.pkl` on the evaluation seed profile (S-MR-002). This student was identified as requiring academic support based on a high predicted disengagement risk (probability = 0.7195, risk score = 71/100). The learner exhibits a Last-Minute Survivor behavioural pattern characterized by inconsistent engagement, high inactivity (160 days), and deadline-driven study habits. The AI Mentor applies an Implementation Intention ("If-Then") planning strategy to encourage better study planning and sustained engagement.

---

## SECTION A — TECHNICAL VERSION (Version A)

### Student Identification

| Field | Value |
|---|---|
| **Anonymized ID** | S-MR-002 |
| **Dataset** | OULAD evaluation seed profile |
| **Profile Key** | `last_minute` |
| **Prediction** | At risk (predicted 1) |

---

### Risk Assessment

| Metric | Value |
|---|---|
| **Predicted Probability** | 0.7195 (71.95%) |
| **Risk Score (0–100)** | 71 / 100 |
| **Risk Level** | 🔴 **High** |
| **Predicted Class** | 1 (At risk) |

**Risk reasons computed by system:**
1. "High inactivity detected (160 days)"
2. "Student engagement trend declining"
3. "Behavior fluctuation increasing"

---

### Behavioural Features (raw values from model input)

| Feature | Value | Risk Direction |
|---|---|---|
| Total Interaction Clicks | 2,275 | Moderate |
| Mean Assessment Score | 50.00 | 🚨 Warning (below-average performance) |
| Active Learning Days | 78 | Moderate |
| Engagement Variability | 20.51 | ⚠️ Warning (elevated fluctuation) |
| Inactivity Duration (days) | 160.0 | 🚨 Warning (very extended absence) |
| Engagement Trend (Slope) | −0.216 | ⚠️ Warning (declining) |
| Assessment Consistency (SD) | 10.90 | Moderate inconsistency |

---

### SHAP Explanation (per-student feature attribution)

**SHAP Base Value:** 0.1206 (baseline risk for an average student)  
**Total SHAP Contribution:** Positive net contribution pushing above the risk threshold

#### Risk-Increasing Features (push toward higher risk):

| Feature | Feature Value | SHAP Value | Effect |
|---|---|---|---|
| Engagement Trend (Slope) | −0.2156 | **+1.555** | Strongly increases risk |

#### Risk-Decreasing Features (protective — push toward lower risk):

| Feature | Feature Value | SHAP Value | Effect |
|---|---|---|---|
| Assessment Consistency (SD) | 10.90 | **−0.480** | Decreases risk |
| Inactivity Duration (days) | 160.0 | **−0.462** | ⚠️ Counterintuitive: despite 160 days absence, the model partially treats this as a "survived the absence" signal |
| Mean Assessment Score | 50.00 | Reduced protective effect | Lower score provides less counterbalance to risk factors |
| Active Learning Days | 78.0 | −0.189 | Decreases risk |
| Total Interaction Clicks | 2,275 | −0.069 | Decreases risk |
| Engagement Variability | 20.51 | −0.052 | Decreases risk |

**Key SHAP insight:** The dominant risk signal is the **declining engagement slope** (SHAP = +1.555), combined with a **below-average assessment score** (50.0) that no longer provides the strong protective counterbalance it did at higher values. The combination pushes the prediction clearly above the risk threshold to 71.95%.

> **SHAP interpretation note on inactivity:** The SHAP value of −0.462 for inactivity at 160 days may seem counterintuitive. In the OULAD dataset, some students with long registration windows have extended periods between module starts. The model has learned that inactivity alone is less predictive than inactivity combined with declining scores. With the current assessment score of 50.0, this partial protective effect is diminished compared to higher-performing students.

---

### Learner Archetype (Persona)

| Field | Value |
|---|---|
| **K-Means Cluster** | Cluster 5 |
| **Archetype Label** | **Last-Minute Survivor** |
| **Intervention Style** | Deadline management coaching |
| **Source** | Frozen `persona_kmeans.pkl` + `advanced_persona_profiles.csv` |

---

### Cognitive State

| Metric | Value | Interpretation |
|---|---|---|
| Attention Score | 85.76 | High attention |
| Confusion Score | 14.24 | Low — well below 70 threshold |
| Boredom Score | 0.0 | No boredom detected |
| Cognitive Overload Score | 68.0 | Elevated — approaching but below 70 threshold |
| Stability Score | 65.75 | Moderate-high behavioural stability |

> *Source: Pre-computed in advanced_persona_profiles.csv.*

---

### Priority Focus (computed by determine_priority_focus())

| Field | Value |
|---|---|
| **Focus ID** | `inactivity` |
| **Instruction to AI** | "Inactivity is the strongest measured risk driver — anchor advice on resuming regular activity, not general study tips." |
| **Reason** | inactivity_days=160.0 (>15 days threshold) |

> *Note: The persona strategy (Last-Minute Survivor) would normally suggest deadline planning. However, the 160-day inactivity threshold triggers the inactivity priority focus instead, which takes precedence over the persona default. The student recommendation still reflects the persona (If-Then planning), but the priority focus anchors it to re-engagement first.*

---

### Confidence Assessment

| Metric | Value |
|---|---|
| **Confidence Level** | **High** |
| **Has Risk Score** | ✅ Yes |
| **Has Persona** | ✅ Yes |
| **Has SHAP** | ✅ Yes |
| **Has Cognitive State** | ✅ Yes |
| **Needs Human Review** | ⚠️ Yes (offline mode; recommended to verify contextual factors behind the high inactivity period) |

**Confidence Rationale:** Risk score, SHAP-based feature explanation, and a known learner persona are all available, giving a quantified, explained, and contextualized basis for this response. Cognitive-state data was also available as a supporting signal.

---

### AI Mentor — Student Recommendation

> **[VERBATIM FROM SYSTEM — `_PERSONA_ACTIONS["Last-Minute Survivor"]["student"]`]**

*"Set a specific If-Then plan (e.g. 'after dinner, review one topic') to space out revision before the deadline."*

---

### AI Mentor — Educator Recommendation

> **[VERBATIM FROM SYSTEM — `_PERSONA_ACTIONS["Last-Minute Survivor"]["educator"]`]**

*"Consider a scaffolded reminder or interim checkpoint ahead of the next deadline."*

---

### Psychological Framework

> **[VERBATIM FROM SYSTEM — `_PSYCHOLOGICAL_FRAMEWORKS["Last-Minute Survivor"]`]**

*"Implementation Intentions — planning: propose a concrete If-Then study micro-habit tied to a specific deadline, recommend spaced revision over cramming, and name the cognitive-load cost of leaving it to the last minute."*

---

### Conversation Strategy

| Element | Value |
|---|---|
| **Opening Goal** | Welcome back, non-judgmental |
| **Teaching Style** | Focus on restarting momentum, not catching up all at once |
| **Recommendation Style** | One tiny piece of coursework today, to break the streak |
| **Educator Goal** | Reach out directly given the inactivity pattern |

---

### Supporting Figures

| Figure | File | Purpose |
|---|---|---|
| **SHAP Waterfall (generated)** | `../ieee_results/figures/eval_waterfall_moderate_risk.png` | Per-student feature attribution for S-MR-002 |
| SHAP Global Importance | `../ieee_results/figures/fig3_shap_importance.png` | Context: global feature rankings |
| SHAP Beeswarm | `../ieee_results/figures/fig4_shap_beeswarm.png` | Distribution across all students |
| Persona Clusters | `../ieee_results/figures/fig_g2_tsne_clusters.png` | Cluster 5 = Last-Minute Survivor |

---

## SECTION B — EDUCATOR VERSION (Version B)

> *Technical explanations rewritten into plain language. AI Mentor recommendations are unchanged — verbatim as produced by the system.*

---

### About This Student

**Student ID:** S-MR-002  
**Risk Level:** 🔴 High Risk  
**AI System Confidence:** High — All required information was available.

---

### What the AI System Found

This student was identified as requiring academic support based on a high predicted disengagement risk (71.95%). Multiple warning signs combine to produce a clear at-risk prediction.

**Some positive indicators:**
- They have logged **2,275 interactions** with course materials, showing some engagement effort.
- When they do engage, their work is **moderately consistent**.

**The main warning signs:**
- **160 days of inactivity** — the student has been absent from the course for an extended period.
- **Declining engagement** — the student's activity level is dropping over time, not staying stable or improving.
- **Fluctuating engagement day-to-day** — their activity is unpredictable, suggesting possible last-minute catch-up behaviour.

---

### Why This Risk Level?

The AI system's explanation shows that multiple factors combine to produce a clear at-risk prediction:

- **Biggest reason pushing toward risk:** The student's engagement is declining (SHAP +1.555). This is the single most important factor.
- **Contributing risk factor:** The student's average assessment score of **50.0** provides significantly less protective counterbalance than a higher score would.
- **160 days of inactivity** further contributes to the elevated risk profile.

**Learner Type:** This student is classified as a **Last-Minute Survivor** — someone who tends to engage sporadically and leave work until close to deadlines.



---

### AI Mentor Recommendation — For the Student

> **Produced by the AI Mentor system. Not paraphrased or edited.**

*"Set a specific If-Then plan (e.g. 'after dinner, review one topic') to space out revision before the deadline."*

---

### AI Mentor Recommendation — For the Educator

> **Produced by the AI Mentor system. Not paraphrased or edited.**

*"Consider a scaffolded reminder or interim checkpoint ahead of the next deadline."*

---

### Why This Recommendation?

The student is identified as a Last-Minute Survivor — someone whose natural tendency is to leave work until deadlines approach. The AI's recommendation is based on Implementation Intentions (a well-researched planning technique): giving people a specific "if X then Y" plan significantly increases the likelihood they'll follow through. The system recommends spaced revision rather than cramming.

The educator recommendation (interim checkpoint) is designed to provide an external structure that compensates for the student's tendency to delay.

---

### Confidence and Human Review

The AI system rated its confidence as **High** — all information was available. **Human educator judgment remains important** to provide context the model cannot access (e.g., the student may have had a personal emergency explaining the 160-day absence, or may have legitimate reasons for the lower assessment scores).
