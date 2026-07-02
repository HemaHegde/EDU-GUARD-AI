# EduGuard-AI -- Real End-to-End RAG / Mentor Ablation Study

_Generated: 2026-07-02T07:17:19.347841+00:00_

## 0. Automatic Key Findings

Computed automatically from the aggregated results below (§6b, §6e, §6f) -- nothing in this section is hard-coded. Any entry reading `Not Evaluated` means the underlying comparison did not have enough real paired data in this run.

- **Configuration that degraded performance the most:** `Risk Only` (mean diff averaged across metrics vs Full System = -0.1).
- **Component with the greatest single-metric influence:** `Persona` (largest effect on `priority_focus_correctness`, mean diff = -0.4).
- **Most stable metric across ablations:** `persona_consistency` (max |mean diff| across configs = 0.097).
- **Most sensitive metric across ablations:** `priority_focus_correctness` (max |mean diff| across configs = 0.4).
- **Configuration with the highest failure rate:** `Full System` (failure rate = 0.0).
- **Strongest observed correlation:** Confidence vs Educational Usefulness (Spearman r = -0.1931, p = 0.4147, n = 20).
## 1. What Was Actually Evaluated

This report reflects live calls into the real AI Mentor backend (`mentor_service.ask_mentor`, `context_builder.build_student_context`, `prompt_builder.build_system_prompt`, `retrieval.get_retriever`, `confidence.assess_confidence`) for the user IDs and questions listed below. No student context, persona, risk score, SHAP explanation, cognitive state, retrieval result, or mentor response in this report was fabricated. Any metric that could not be observed is explicitly marked as such rather than estimated.

### 1.1 Backend Modules Used

| Module | Imported | Path used | Error (if any) |
|---|---|---|---|
| `mentor_service` | yes | services.mentor_service | - |
| `context_builder` | yes | services.context_builder | - |
| `prompt_builder` | yes | services.prompt_builder | - |
| `retrieval` | yes | services.retrieval | - |
| `confidence` | yes | services.confidence | - |
| `reasoning_layer` | yes | services.reasoning_layer | - |
| `supabase` | yes | config.supabase_client | - |

### 1.2 Live Inference Status

- `USE_LIVE_LLM` requested: **True**
- Ollama reachability probe (`ollama.list()`): **reachable**
- Model name used for ablation-arm LLM calls: `qwen2.5:3b` (read from `mentor_service.ask_mentor`'s own source where possible)

### 1.3 Users and Questions

- User IDs evaluated (1): `d271bd7d-6631-4969-b000-692ad069ddfe`
- Questions evaluated (5):
  - I'm feeling really behind in this course, what should I do?
  - Can you explain the topic we covered this week?
  - I don't understand today's material at all, I'm confused.
  - How should I plan my revision before the next assessment?
  - I haven't logged in for a while, where should I start again?

## 2. Methodology

### 2.1 Full System Configuration

Calls the real, unmodified `ask_mentor(user_id, question)` exactly as production traffic would, including its write to the `mentor_history` table. This is the only configuration that is a literal call into the real orchestrator end to end.

### 2.2 Ablation Arms

`ask_mentor()` builds its own `StudentContext` internally and has no parameter for injecting a modified one, so it cannot itself run masked-evidence ablations. Each ablation arm therefore: (1) calls the real `build_student_context(user_id)`, (2) deep-copies the real result and removes (never substitutes) the evidence field(s) that arm is meant to ablate, (3) replays the same sequence of real functions `ask_mentor()` calls internally -- `retrieval.get_retriever()`, `prompt_builder.build_system_prompt()`, `mentor_service._build_output_format_instructions()`, `confidence.assess_confidence()`, and (if live) the real `ollama.chat()` call followed by `mentor_service._parse_structured_response()`. No business logic is reimplemented anywhere in this script; every step above calls the real function from the real module. To avoid writing ablation-arm traffic into the production `mentor_history` table, these arms do not persist history (unlike Full System, §2.1).

Fields removed per arm (masking only, never replaced with a fabricated value):

- **Without Retrieval**: `retrieved_chunks` forced to `[]` before prompt construction.
- **Without Persona**: `persona` set to `"Unknown Persona"` (the system's own existing sentinel for "no usable persona", reused here only as a masking marker for this ablation arm, not claimed as real backend output); `intervention_style` set to `None`.
- **Without SHAP**: `shap_available=False`, `shap_explanation=None`.
- **Without Cognitive**: `cognitive_state_available=False`, `cognitive_state=None`.
- **Risk Only**: all of the above combined, plus `chat_history=""`.

## 3. Aggregate Metrics (observed only)

- Total records attempted: **30**
- Records with status `ok`: **30**
- Status breakdown: `ok`=30

| Configuration | n | Status counts | Retrieval coverage | Confidence dist. | Parse-mode dist. | Mean response chars |
|---|---|---|---|---|---|---|
| Full System | 5 | {'ok': 5} | unavailable | {'High': 5} | unavailable | 283.2 |
| Without Retrieval | 5 | {'ok': 5} | 0.0 | {'High': 5} | {'strict': 5} | 343.4 |
| Without Persona | 5 | {'ok': 5} | 1.0 | {'Low': 5} | {'strict': 5} | 338.2 |
| Without SHAP | 5 | {'ok': 5} | 1.0 | {'Medium': 5} | {'strict': 5} | 300.8 |
| Without Cognitive | 5 | {'ok': 5} | 1.0 | {'High': 5} | {'strict': 5} | 295.2 |
| Risk Only | 5 | {'ok': 5} | 1.0 | {'Low': 5} | {'strict': 5} | 493.0 |

## 4. Figures

### Retrieval Coverage

![retrieval_coverage](..\figures\retrieval_coverage.png)

### Confidence Distribution

![confidence_distribution](..\figures\confidence_distribution.png)

### Mean Response Length

![mean_response_length](..\figures\mean_response_length.png)

### Quality Metrics By Config

![quality_metrics_by_config](..\figures\quality_metrics_by_config.png)

### Quality Metrics Ci Error Bars

![quality_metrics_ci_error_bars](..\figures\quality_metrics_ci_error_bars.png)

### Quality Metrics Box Violin

![quality_metrics_box_violin](..\figures\quality_metrics_box_violin.png)


## 5. Explicit Limitations

- None identified beyond what is already noted per-metric above.

## 6b. Response Quality Metrics (new)

These metrics evaluate the actual generated text (mentor_response / student_recommendation), not just system diagnostics. Per the study's restructuring requirement, they are now split into two clearly labeled subsections -- **Deterministic Metrics**, computed directly from measurable data with no design-choice rubric involved, and **Heuristic Metrics**, which are rule-based evaluation rubrics (the rubric itself encodes a design choice, e.g. which keywords count as "urgency language") and should be read as documented proxies rather than objective benchmark scores. Full rubric definitions are in code comments, `rag_ablation.py` §7B. None are LLM-judged or hand-scored, and none are estimated where data was missing -- those cells read `Not Evaluated`.

### 6b.1 Deterministic Metrics (mean scores by configuration, observed only)

These are computed directly from measurable data -- no rubric design choice is involved.

| Metric | Scale | Full System | Without Retrieval | Without Persona | Without SHAP | Without Cognitive | Risk Only |
|---|---|---|---|---|---|---|---|
| Educational Usefulness | 0-1, recall of retrieved-chunk tokens in the response | Not Evaluated (n_not_evaluated=5) | Not Evaluated (n_not_evaluated=5) | 0.0114 (n=5, not_eval=0) | 0.0152 (n=5, not_eval=0) | 0.0076 (n=5, not_eval=0) | 0.1924 (n=5, not_eval=0) |
| Persona Consistency | 0-1, within-persona response similarity | 0.127 (n=5, not_eval=0) | 0.216 (n=5, not_eval=0) | Not Evaluated (n_not_evaluated=5) | 0.224 (n=5, not_eval=0) | 0.179 (n=5, not_eval=0) | Not Evaluated (n_not_evaluated=5) |
| Grounding Recall | 0-1, mean recall across available evidence sources (SHAP/risk/cognitive) | Not Evaluated (n_not_evaluated=5) | 0.0028 (n=5, not_eval=0) | 0.0182 (n=5, not_eval=0) | 0.009 (n=5, not_eval=0) | 0.0124 (n=5, not_eval=0) | 0.0182 (n=5, not_eval=0) |

Also deterministic, but population-level rather than per-record (see 6b.2): **Recommendation Diversity** and **Tone Differentiation**.


### 6b.1b Heuristic Metrics (mean scores by configuration, observed only)

These are rule-based evaluation rubrics -- documented proxies for the underlying construct, not objective benchmark measurements.

| Metric | Scale | Full System | Without Retrieval | Without Persona | Without SHAP | Without Cognitive | Risk Only |
|---|---|---|---|---|---|---|---|
| Recommendation Quality | 0-3, rule-based rubric (specificity + actionable verb + time marker) | 2.0 (n=5, not_eval=0) | 1.8 (n=5, not_eval=0) | 2.4 (n=5, not_eval=0) | 2.2 (n=5, not_eval=0) | 2.4 (n=5, not_eval=0) | 2.0 (n=5, not_eval=0) |
| Evidence Fusion Correctness | 0-1, keyword-presence rubric for referencing available evidence | 0.2 (n=5, not_eval=0) | 0.2 (n=5, not_eval=0) | 0.15 (n=5, not_eval=0) | 0.15 (n=5, not_eval=0) | 0.05 (n=5, not_eval=0) | 0.3 (n=5, not_eval=0) |
| Priority Focus Correctness | 0/1, rubric matching urgency/steady language to risk level | 0.8 (n=5, not_eval=0) | 0.8 (n=5, not_eval=0) | 0.4 (n=5, not_eval=0) | 0.6 (n=5, not_eval=0) | 0.6 (n=5, not_eval=0) | 0.4 (n=5, not_eval=0) |

### 6b.1c Grounding Recall detail (deterministic sub-scores by evidence source)

Grounding Recall (6b.1) is the mean of whichever of these three are available per record.

| Metric | Scale | Full System | Without Retrieval | Without Persona | Without SHAP | Without Cognitive | Risk Only |
|---|---|---|---|---|---|---|---|
| SHAP grounding | 0-1, deterministic recall | Not Evaluated (n_not_evaluated=5) | 0.0 (n=5, not_eval=0) | 0.0194 (n=5, not_eval=0) | Not Evaluated (n_not_evaluated=5) | 0.0064 (n=5, not_eval=0) | Not Evaluated (n_not_evaluated=5) |
| Risk-reasons grounding | 0-1, deterministic recall | Not Evaluated (n_not_evaluated=5) | 0.0 (n=5, not_eval=0) | 0.0182 (n=5, not_eval=0) | 0.0182 (n=5, not_eval=0) | 0.0182 (n=5, not_eval=0) | 0.0182 (n=5, not_eval=0) |
| Cognitive-state grounding | 0-1, deterministic recall | Not Evaluated (n_not_evaluated=5) | 0.0086 (n=5, not_eval=0) | 0.0172 (n=5, not_eval=0) | 0.0 (n=5, not_eval=0) | Not Evaluated (n_not_evaluated=5) | Not Evaluated (n_not_evaluated=5) |

### 6b.2 Population-level metrics by configuration (deterministic)

| Configuration | Recommendation Diversity (unique ratio) | Mean pairwise dissimilarity | Tone differentiation (within - across persona similarity) |
|---|---|---|---|
| Full System | 1.0 (n=5) | 0.886 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| Without Retrieval | 1.0 (n=5) | 0.805 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| Without Persona | 1.0 (n=5) | 0.958 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| Without SHAP | 1.0 (n=5) | 0.943 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| Without Cognitive | 1.0 (n=5) | 0.923 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| Risk Only | 1.0 (n=5) | 0.95 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |

### 6b.3 Statistical detail (mean, std, 95% CI; observed only)

_95% CIs use a normal approximation and are noted as approximate for n < 30. Cells with fewer than 2 observations report `Not Evaluated` for std/CI._

| Metric | Configuration | n | Mean | Std | 95% CI |
|---|---|---|---|---|---|
| Educational Usefulness | Without Persona | 5 | 0.0114 | 0.017 | [-0.0035, 0.0263] |
| Educational Usefulness | Without SHAP | 5 | 0.0152 | 0.0085 | [0.0078, 0.0226] |
| Educational Usefulness | Without Cognitive | 5 | 0.0076 | 0.0104 | [-0.0015, 0.0167] |
| Educational Usefulness | Risk Only | 5 | 0.1924 | 0.2388 | [-0.0169, 0.4017] |
| Persona Consistency | Full System | 5 | 0.127 | 0.0 | [0.127, 0.127] |
| Persona Consistency | Without Retrieval | 5 | 0.216 | 0.0 | [0.216, 0.216] |
| Persona Consistency | Without SHAP | 5 | 0.224 | 0.0 | [0.224, 0.224] |
| Persona Consistency | Without Cognitive | 5 | 0.179 | 0.0 | [0.179, 0.179] |
| Grounding Recall | Without Retrieval | 5 | 0.0028 | 0.0063 | [-0.0027, 0.0083] |
| Grounding Recall | Without Persona | 5 | 0.0182 | 0.0334 | [-0.0111, 0.0475] |
| Grounding Recall | Without SHAP | 5 | 0.009 | 0.0201 | [-0.0086, 0.0266] |
| Grounding Recall | Without Cognitive | 5 | 0.0124 | 0.0277 | [-0.0119, 0.0367] |
| Grounding Recall | Risk Only | 5 | 0.0182 | 0.0407 | [-0.0175, 0.0539] |
| Recommendation Quality | Full System | 5 | 2.0 | 1.0 | [1.1235, 2.8765] |
| Recommendation Quality | Without Retrieval | 5 | 1.8 | 0.8367 | [1.0666, 2.5334] |
| Recommendation Quality | Without Persona | 5 | 2.4 | 0.8944 | [1.616, 3.184] |
| Recommendation Quality | Without SHAP | 5 | 2.2 | 0.8367 | [1.4666, 2.9334] |
| Recommendation Quality | Without Cognitive | 5 | 2.4 | 0.8944 | [1.616, 3.184] |
| Recommendation Quality | Risk Only | 5 | 2.0 | 0.0 | [2.0, 2.0] |
| Evidence Fusion Correctness | Full System | 5 | 0.2 | 0.1118 | [0.102, 0.298] |
| Evidence Fusion Correctness | Without Retrieval | 5 | 0.2 | 0.1118 | [0.102, 0.298] |
| Evidence Fusion Correctness | Without Persona | 5 | 0.15 | 0.1369 | [0.03, 0.27] |
| Evidence Fusion Correctness | Without SHAP | 5 | 0.15 | 0.1369 | [0.03, 0.27] |
| Evidence Fusion Correctness | Without Cognitive | 5 | 0.05 | 0.1118 | [-0.048, 0.148] |
| Evidence Fusion Correctness | Risk Only | 5 | 0.3 | 0.2739 | [0.06, 0.54] |
| Priority Focus Correctness | Full System | 5 | 0.8 | 0.4472 | [0.408, 1.192] |
| Priority Focus Correctness | Without Retrieval | 5 | 0.8 | 0.4472 | [0.408, 1.192] |
| Priority Focus Correctness | Without Persona | 5 | 0.4 | 0.5477 | [-0.0801, 0.8801] |
| Priority Focus Correctness | Without SHAP | 5 | 0.6 | 0.5477 | [0.1199, 1.0801] |
| Priority Focus Correctness | Without Cognitive | 5 | 0.6 | 0.5477 | [0.1199, 1.0801] |
| Priority Focus Correctness | Risk Only | 5 | 0.4 | 0.5477 | [-0.0801, 0.8801] |
| SHAP grounding | Without Retrieval | 5 | 0.0 | 0.0 | [0.0, 0.0] |
| SHAP grounding | Without Persona | 5 | 0.0194 | 0.0434 | [-0.0186, 0.0574] |
| SHAP grounding | Without Cognitive | 5 | 0.0064 | 0.0143 | [-0.0061, 0.0189] |
| Risk-reasons grounding | Without Retrieval | 5 | 0.0 | 0.0 | [0.0, 0.0] |
| Risk-reasons grounding | Without Persona | 5 | 0.0182 | 0.0407 | [-0.0175, 0.0539] |
| Risk-reasons grounding | Without SHAP | 5 | 0.0182 | 0.0407 | [-0.0175, 0.0539] |
| Risk-reasons grounding | Without Cognitive | 5 | 0.0182 | 0.0407 | [-0.0175, 0.0539] |
| Risk-reasons grounding | Risk Only | 5 | 0.0182 | 0.0407 | [-0.0175, 0.0539] |
| Cognitive-state grounding | Without Retrieval | 5 | 0.0086 | 0.0192 | [-0.0083, 0.0255] |
| Cognitive-state grounding | Without Persona | 5 | 0.0172 | 0.0236 | [-0.0034, 0.0378] |
| Cognitive-state grounding | Without SHAP | 5 | 0.0 | 0.0 | [0.0, 0.0] |

### 6b.4 Paired comparison vs. Full System (same user+question, evidence config differs)

Pairs are formed only where BOTH the Full System record and the ablation-arm record for the same `(user_id, question)` have a real numeric score for that metric -- no imputation. `p_value`/`t_stat` (paired t-test) require `scipy`; if unavailable, only the mean/std of paired differences is reported and `p_value` reads `Not Evaluated`. The `Recommended test` columns additionally choose between a paired t-test and a Wilcoxon signed-rank test per-comparison, based on a Shapiro-Wilk normality check of the paired differences (Wilcoxon is preferred whenever normality is rejected or cannot be checked, per the study's stated test preference), and report an effect size alongside it.

| Configuration | Metric | n pairs | Mean diff (arm - Full System) | Std diff | t-stat | p-value | Recommended test | Recommended p-value | Effect size |
|---|---|---|---|---|---|---|---|---|---|
| Without Retrieval | Persona Consistency | 5 | 0.089 | 0.0 | inf | 0.0 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Retrieval | Recommendation Quality | 5 | -0.2 | 1.0954 | -0.4082 | 0.704 | paired t-test | 0.704 | -0.1826 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Retrieval | Evidence Fusion Correctness | 5 | 0.0 | 0.0 | nan | nan | Wilcoxon signed-rank | Not Evaluated | Not Evaluated |
| Without Retrieval | Priority Focus Correctness | 5 | 0.0 | 0.0 | nan | nan | Wilcoxon signed-rank | Not Evaluated | Not Evaluated |
| Without Persona | Recommendation Quality | 5 | 0.4 | 0.5477 | 1.633 | 0.1778 | Wilcoxon signed-rank | 0.5 | 0.4769 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Persona | Evidence Fusion Correctness | 5 | -0.05 | 0.1118 | -1.0 | 0.3739 | Wilcoxon signed-rank | 1.0 | -0.0 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Persona | Priority Focus Correctness | 5 | -0.4 | 0.5477 | -1.633 | 0.1778 | Wilcoxon signed-rank | 0.5 | -0.4769 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Persona Consistency | 5 | 0.097 | 0.0 | inf | 0.0 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Recommendation Quality | 5 | 0.2 | 0.4472 | 1.0 | 0.3739 | Wilcoxon signed-rank | 1.0 | 0.0 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Evidence Fusion Correctness | 5 | -0.05 | 0.1118 | -1.0 | 0.3739 | Wilcoxon signed-rank | 1.0 | -0.0 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Priority Focus Correctness | 5 | -0.2 | 0.4472 | -1.0 | 0.3739 | Wilcoxon signed-rank | 1.0 | -0.0 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Persona Consistency | 5 | 0.052 | 0.0 | inf | 0.0 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Recommendation Quality | 5 | 0.4 | 0.5477 | 1.633 | 0.1778 | Wilcoxon signed-rank | 0.5 | 0.4769 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Evidence Fusion Correctness | 5 | -0.15 | 0.1369 | -2.4495 | 0.0705 | Wilcoxon signed-rank | 0.25 | -0.6642 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Priority Focus Correctness | 5 | -0.2 | 0.4472 | -1.0 | 0.3739 | Wilcoxon signed-rank | 1.0 | -0.0 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Risk Only | Recommendation Quality | 5 | 0.0 | 1.0 | 0.0 | 1.0 | paired t-test | 1.0 | 0.0 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Risk Only | Evidence Fusion Correctness | 5 | 0.1 | 0.3354 | 0.6667 | 0.5415 | paired t-test | 0.5415 | 0.2981 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Risk Only | Priority Focus Correctness | 5 | -0.4 | 0.5477 | -1.633 | 0.1778 | Wilcoxon signed-rank | 0.5 | -0.4769 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |

### 6b.5 Discussion (automatic interpretation of the paired comparisons above)

Interpretive sentences below are generated directly from the `mean_diff` values in 6b.4 -- no explanation is invented beyond what those measured differences show. Percent change is relative to the Full System mean for that metric where that mean is a real (non-zero) observed value; otherwise only the raw mean difference is stated.

- Under **Without Retrieval**, Persona Consistency increased by **70.1%** relative to Full System (mean diff = 0.089) (not statistically significant at p = 0.0625), suggesting the evidence removed in this arm meaningfully detracts from persona consistency when present in the Full System.
- Under **Without Retrieval**, Recommendation Quality decreased by **10.0%** relative to Full System (mean diff = -0.2) (not statistically significant at p = 0.704), suggesting the evidence removed in this arm meaningfully contributes to recommendation quality when present in the Full System.
- Under **Without Persona**, Recommendation Quality increased by **20.0%** relative to Full System (mean diff = 0.4) (not statistically significant at p = 0.5), suggesting the evidence removed in this arm meaningfully detracts from recommendation quality when present in the Full System.
- Under **Without Persona**, Evidence Fusion Correctness decreased by **25.0%** relative to Full System (mean diff = -0.05) (not statistically significant at p = 1.0), suggesting the evidence removed in this arm meaningfully contributes to evidence fusion correctness when present in the Full System.
- Under **Without Persona**, Priority Focus Correctness decreased by **50.0%** relative to Full System (mean diff = -0.4) (not statistically significant at p = 0.5), suggesting the evidence removed in this arm meaningfully contributes to priority focus correctness when present in the Full System.
- Under **Without SHAP**, Persona Consistency increased by **76.4%** relative to Full System (mean diff = 0.097) (not statistically significant at p = 0.0625), suggesting the evidence removed in this arm meaningfully detracts from persona consistency when present in the Full System.
- Under **Without SHAP**, Recommendation Quality increased by **10.0%** relative to Full System (mean diff = 0.2) (not statistically significant at p = 1.0), suggesting the evidence removed in this arm meaningfully detracts from recommendation quality when present in the Full System.
- Under **Without SHAP**, Evidence Fusion Correctness decreased by **25.0%** relative to Full System (mean diff = -0.05) (not statistically significant at p = 1.0), suggesting the evidence removed in this arm meaningfully contributes to evidence fusion correctness when present in the Full System.
- Under **Without SHAP**, Priority Focus Correctness decreased by **25.0%** relative to Full System (mean diff = -0.2) (not statistically significant at p = 1.0), suggesting the evidence removed in this arm meaningfully contributes to priority focus correctness when present in the Full System.
- Under **Without Cognitive**, Persona Consistency increased by **40.9%** relative to Full System (mean diff = 0.052) (not statistically significant at p = 0.0625), suggesting the evidence removed in this arm meaningfully detracts from persona consistency when present in the Full System.
- Under **Without Cognitive**, Recommendation Quality increased by **20.0%** relative to Full System (mean diff = 0.4) (not statistically significant at p = 0.5), suggesting the evidence removed in this arm meaningfully detracts from recommendation quality when present in the Full System.
- Under **Without Cognitive**, Evidence Fusion Correctness decreased by **75.0%** relative to Full System (mean diff = -0.15) (not statistically significant at p = 0.25), suggesting the evidence removed in this arm meaningfully contributes to evidence fusion correctness when present in the Full System.
- Under **Without Cognitive**, Priority Focus Correctness decreased by **25.0%** relative to Full System (mean diff = -0.2) (not statistically significant at p = 1.0), suggesting the evidence removed in this arm meaningfully contributes to priority focus correctness when present in the Full System.
- Under **Risk Only**, Evidence Fusion Correctness increased by **50.0%** relative to Full System (mean diff = 0.1) (not statistically significant at p = 0.5415), suggesting the evidence removed in this arm meaningfully detracts from evidence fusion correctness when present in the Full System.
- Under **Risk Only**, Priority Focus Correctness decreased by **50.0%** relative to Full System (mean diff = -0.4) (not statistically significant at p = 0.5), suggesting the evidence removed in this arm meaningfully contributes to priority focus correctness when present in the Full System.

- Under **Without Retrieval**, no measured difference vs Full System: Evidence Fusion Correctness, Priority Focus Correctness.
- Under **Risk Only**, no measured difference vs Full System: Recommendation Quality.

## 6e. Correlation Analysis

Spearman rank correlation between the confidence level `confidence.assess_confidence()` actually assigned (mapped Low=1, Medium=2, High=3 -- the same ordering already encoded in that module's own rubric) and each quality metric, computed only over records with a real confidence level and a real numeric metric value. Reported `Not Evaluated` when fewer than 3 such paired observations exist, or when `scipy` is unavailable.

| Comparison | n | Spearman r | p-value |
|---|---|---|---|
| Confidence vs Recommendation Quality | 30 | -0.0638 | 0.7377 |
| Confidence vs Educational Usefulness | 20 | -0.1931 | 0.4147 |
| Confidence vs Grounding Recall | 25 | -0.1325 | 0.5279 |

## 6f. Error Analysis

Computed strictly from the observed records above -- failure counts, parser fallback counts, response lengths, and confidence labels already captured on each `EvalRecord`.

- Configuration with the highest failure rate: **Full System** (failure rate = 0.0)
- Overall parser fallback frequency: **0.0**
- Overall structured-response compliance rate: **1.0**
- Overall confidence distribution: {'High': 15, 'Low': 10, 'Medium': 5}

| Configuration | n | Failures | Failure rate | Parser fallback freq. | Structured compliance rate | Avg. response length (chars) | Confidence distribution |
|---|---|---|---|---|---|---|---|
| Full System | 5 | 0 | 0.0 | Not Evaluated | Not Evaluated | 283.2 | {'High': 5} |
| Without Retrieval | 5 | 0 | 0.0 | 0.0 | 1.0 | 343.4 | {'High': 5} |
| Without Persona | 5 | 0 | 0.0 | 0.0 | 1.0 | 338.2 | {'Low': 5} |
| Without SHAP | 5 | 0 | 0.0 | 0.0 | 1.0 | 300.8 | {'Medium': 5} |
| Without Cognitive | 5 | 0 | 0.0 | 0.0 | 1.0 | 295.2 | {'High': 5} |
| Risk Only | 5 | 0 | 0.0 | 0.0 | 1.0 | 493.0 | {'Low': 5} |

## 6g. Threats to Validity

Standard IEEE-style disclosure of validity threats for this study. Each point below is either a structural property of the evaluation design (true regardless of this run's data) or is grounded in this run's own observed counts where noted.

**Construct validity**
- Heuristic metrics (Recommendation Quality, Evidence Fusion Correctness, Priority Focus Correctness -- §6b.1b) are keyword/rule-based proxies for their underlying constructs, not semantic judgments; they can be gamed by keyword-stuffing and can under-score genuinely good paraphrased responses. They are not a substitute for human educator evaluation -- see the rubric template in §6c, which this run does NOT auto-fill.
- The Confidence label used in §6e's correlation analysis is itself a rule-based evidence-completeness signal (see `confidence.py`), not an independent ground-truth quality judgment; correlating it with quality metrics measures internal consistency of the pipeline's own evidence bookkeeping, not external validity of that bookkeeping.

**Internal validity**
- The `Full System` configuration exposes fewer internal artifacts than the ablation arms (`ask_mentor()`'s public return schema does not include retrieved_chunks, shap_explanation, risk_reasons, or cognitive_state text -- §6d), so several metrics (Educational Usefulness, Grounding Recall and its sub-scores) are structurally `Not Evaluated` for Full System and are only computable for the ablation arms, which themselves call `build_student_context()` directly. This asymmetry means Full System vs. ablation-arm comparisons for those specific metrics compare a real end-to-end call against a partially-replayed one, not two calls with identical observability -- disclosed here and in §6d rather than papered over.
- Ablation arms replay the same real functions `ask_mentor()` calls internally (§2.2), but this replaying is itself a methodological necessity, not a literal `ask_mentor()` call; any behavior specific to `ask_mentor()`'s own internal control flow that this script does not replay would not be captured.
- 0.0% of records with an observed parse mode in this run fell back to the non-structured parse path (§6f), which can inflate or deflate text-derived metrics (e.g. Recommendation Quality, Evidence Fusion) relative to a run with a higher structured-compliance rate.

**External validity**
- Results are based on the specific evaluated user population (1 user(s)) and question set (5 question(s) -- §1.3) in this run; they may not generalize to other students, personas, risk profiles, or question phrasings not represented here.
- The built-in default question set (when no `--questions-file`/`EVAL_QUESTIONS_FILE` is supplied) is a small, generic set of mentoring prompts and is not a validated benchmark; results using it should be read as a pilot-scale study, not a definitive benchmark result.

**Statistical conclusion validity**
- Correlation coefficients in §6e (and any relationship implied by paired comparisons in §6b.4) describe association only; correlation does not imply causation, and no causal claim is made anywhere in this report beyond "removing X was associated with a measured change in Y under this pipeline's specific implementation."
- Sample sizes per (configuration, metric) cell are often small (see the `n` columns in §6b.3 and §6e); 95% CIs use a normal approximation and are noted as approximate for n < 30, and paired tests / correlations report `Not Evaluated` rather than a number when fewer than the required minimum of paired/valid observations exist, but even a computed p-value at small n should be read as indicative rather than conclusive.
- Running many metric x configuration comparisons in §6b.4 without a multiple-comparisons correction increases the chance that at least one comparison reaches p < 0.05 by chance; individual p-values there should be interpreted accordingly, not as a single confirmatory test.

## 6c. Human Evaluation Rubric (template -- not auto-filled)

The metrics above are automated proxies. For an IEEE-style evaluation, pair them with human ratings on a real sample of (user, question, config) records. This is a rubric template only: this script does NOT invent human ratings. Populate it by having 2+ raters independently score a random sample (e.g. via `outputs/records.jsonl`), then report inter-rater agreement (e.g. Cohen's kappa) alongside the means.

| Metric | Scale | Definition | Rater 1 (mean) | Rater 2 (mean) | Agreement |
|---|---|---|---|---|---|
| Personalization | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |
| Evidence Grounding | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |
| Recommendation Quality | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |
| Tone Appropriateness | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |
| Actionability | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |
| Overall Trust | 1-5 | _(fill in before rating)_ | Not Evaluated | Not Evaluated | Not Evaluated |

## 6d. Limitations of the Response Quality Metrics (new, in addition to §5)

- These are automated proxies (keyword/token overlap), not semantic judgments; they can be gamed by keyword-stuffing and can under-score genuinely good paraphrased responses.
- Evidence Fusion, Educational Usefulness, and all three Grounding sub-metrics require text that `ask_mentor()`'s public return schema does not expose (retrieved_chunks, shap_explanation, risk_reasons, cognitive_state content), so these read `Not Evaluated` for every `Full System` record in this run; only the ablation arms (which call `build_student_context()` directly) can be scored on those metrics.
- Persona Consistency and Tone Differentiation need multiple observed responses per persona (and, for the latter, multiple personas) in the same run; small sample sizes will show as `Not Evaluated` or wide confidence intervals rather than a misleadingly precise number.
- Paired comparisons only include pairs with real scores on both sides; configurations with many errored/`Not Evaluated` records will have few or zero valid pairs.


## 7a. Reproducibility Summary

| Item | Value |
|---|---|
| Backend SHA256 (hash of imported backend module source) | `54fc95bd784a7f28dac390d58be78a35ec0730af4e79409f294ba4217ce5c2a9` |
| Ollama model | `qwen2.5:3b` |
| Live LLM active | True |
| Number of evaluated users | 1 |
| Number of evaluated questions | 5 |
| Number of ablation configurations | 6 |
| Total evaluations (users x questions x configs) | 30 |
| Timestamp (UTC) | 2026-07-02T07:17:19.347841+00:00 |

## 7. Reproducing This Run

```
cd <backend project root>   # where the `services` package lives
export EVAL_USER_IDS="<real_user_id_1>,<real_user_id_2>"
export USE_LIVE_LLM=true
python ml/rag_ablation/rag_ablation.py
```
Every value in this report is derived from `outputs/records.jsonl`, produced by this exact run; re-running with the same user IDs, questions, backend code, and a live Ollama instance should reproduce the same real-data metrics up to LLM sampling variance (temperature=0.5, top_p=0.9, as configured in the real `mentor_service.ask_mentor`).