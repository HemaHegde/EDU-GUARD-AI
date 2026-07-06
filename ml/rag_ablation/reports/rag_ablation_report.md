# EduGuard-AI -- Real End-to-End RAG / Mentor Ablation Study

_Generated: 2026-07-06T11:22:14.672838+00:00_

## 0. Automatic Key Findings

Computed automatically from the aggregated results below (§6b, §6e, §6f) -- nothing in this section is hard-coded. Any entry reading `Not Evaluated` means the underlying comparison did not have enough real paired data in this run.

- **Configuration that degraded performance the most:** `Without SHAP` (mean diff averaged across metrics vs Full System = 0.0406).
- **Component with the greatest single-metric influence:** `Persona` (largest effect on `recommendation_quality`, mean diff = 1.0).
- **Most stable metric across ablations:** `bleu_score` (max |mean diff| across configs = 0.0089).
- **Most sensitive metric across ablations:** `recommendation_quality` (max |mean diff| across configs = 1.0).
- **Configuration with the highest failure rate:** `Full System` (failure rate = 0.0).
- **Strongest observed correlation:** Confidence vs Educational Usefulness (Spearman r = -0.4179, p = 0.0073, n = 40).
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

- User IDs evaluated (2): `d271bd7d-6631-4969-b000-692ad069ddfe, 91987f3d-9206-4597-b1fc-a21265f5c673`
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

- Total records attempted: **70**
- Records with status `ok`: **70**
- Status breakdown: `ok`=70

| Configuration | n | Status counts | Retrieval coverage | Confidence dist. | Parse-mode dist. | Mean response chars |
|---|---|---|---|---|---|---|
| Full System | 10 | {'ok': 10} | unavailable | {'High': 10} | unavailable | 92.0 |
| Without Retrieval | 10 | {'ok': 10} | 0.0 | {'High': 10} | {'strict': 9, 'lenient': 1} | 234.5 |
| Without Persona | 10 | {'ok': 10} | 1.0 | {'Low': 10} | {'strict': 10} | 335.1 |
| Without SHAP | 10 | {'ok': 10} | 1.0 | {'Medium': 10} | {'strict': 10} | 256.8 |
| Without Cognitive | 10 | {'ok': 10} | 1.0 | {'High': 10} | {'strict': 10} | 274.5 |
| Risk Only | 10 | {'ok': 10} | 1.0 | {'Low': 10} | {'strict': 10} | 316.1 |
| LLM Only | 10 | {'ok': 10} | 0.0 | unavailable | {'strict': 10} | 313.5 |

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

### Bleu Comparison

![bleu_comparison](..\figures\bleu_comparison.png)

### Semantic Similarity Comparison

![semantic_similarity_comparison](..\figures\semantic_similarity_comparison.png)

### Llm Only Comparison

![llm_only_comparison](..\figures\llm_only_comparison.png)


## 5. Explicit Limitations

- None identified beyond what is already noted per-metric above.

## 6b. Response Quality Metrics (new)

These metrics evaluate the actual generated text (mentor_response / student_recommendation), not just system diagnostics. Per the study's restructuring requirement, they are now split into two clearly labeled subsections -- **Deterministic Metrics**, computed directly from measurable data with no design-choice rubric involved, and **Heuristic Metrics**, which are rule-based evaluation rubrics (the rubric itself encodes a design choice, e.g. which keywords count as "urgency language") and should be read as documented proxies rather than objective benchmark scores. Full rubric definitions are in code comments, `rag_ablation.py` §7B. None are LLM-judged or hand-scored, and none are estimated where data was missing -- those cells read `Not Evaluated`.

### 6b.1 Deterministic Metrics (mean scores by configuration, observed only)

These are computed directly from measurable data -- no rubric design choice is involved.

| Metric | Scale | Full System | Without Retrieval | Without Persona | Without SHAP | Without Cognitive | Risk Only | LLM Only |
|---|---|---|---|---|---|---|---|---|
| Educational Usefulness | 0-1, recall of retrieved-chunk tokens in the response | Not Evaluated (n_not_evaluated=10) | Not Evaluated (n_not_evaluated=10) | 0.0849 (n=10, not_eval=0) | 0.0397 (n=10, not_eval=0) | 0.0019 (n=10, not_eval=0) | 0.0679 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) |
| Persona Consistency | 0-1, within-persona response similarity | 0.6 (n=10, not_eval=0) | 0.1115 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) | 0.1125 (n=10, not_eval=0) | 0.103 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) | Not Evaluated (n_not_evaluated=10) |
| Grounding Recall | 0-1, mean recall across available evidence sources (SHAP/risk/cognitive) | Not Evaluated (n_not_evaluated=10) | 0.0093 (n=10, not_eval=0) | 0.0117 (n=10, not_eval=0) | 0.0095 (n=10, not_eval=0) | 0.0188 (n=10, not_eval=0) | 0.0091 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) |
| BLEU Score | 0-1, sentence-level BLEU (nltk, smoothed) vs. gold reference | 0.0004 (n=10, not_eval=0) | 0.0093 (n=10, not_eval=0) | 0.0049 (n=10, not_eval=0) | 0.0042 (n=10, not_eval=0) | 0.0043 (n=10, not_eval=0) | 0.0073 (n=10, not_eval=0) | 0.0051 (n=10, not_eval=0) |
| Semantic Similarity | 0-1, cosine similarity of all-MiniLM-L6-v2 embeddings vs. gold reference | 0.0925 (n=10, not_eval=0) | 0.4453 (n=10, not_eval=0) | 0.4178 (n=10, not_eval=0) | 0.4197 (n=10, not_eval=0) | 0.4338 (n=10, not_eval=0) | 0.4465 (n=10, not_eval=0) | 0.548 (n=10, not_eval=0) |

Also deterministic, but population-level rather than per-record (see 6b.2): **Recommendation Diversity** and **Tone Differentiation**.


### 6b.1b Heuristic Metrics (mean scores by configuration, observed only)

These are rule-based evaluation rubrics -- documented proxies for the underlying construct, not objective benchmark measurements.

| Metric | Scale | Full System | Without Retrieval | Without Persona | Without SHAP | Without Cognitive | Risk Only | LLM Only |
|---|---|---|---|---|---|---|---|---|
| Recommendation Quality | 0-3, rule-based rubric (specificity + actionable verb + time marker) | 1.9 (n=10, not_eval=0) | 1.9 (n=10, not_eval=0) | 2.9 (n=10, not_eval=0) | 1.7 (n=10, not_eval=0) | 2.0 (n=10, not_eval=0) | 2.4 (n=10, not_eval=0) | 1.9 (n=10, not_eval=0) |
| Evidence Fusion Correctness | 0-1, keyword-presence rubric for referencing available evidence | 0.0 (n=10, not_eval=0) | 0.2 (n=10, not_eval=0) | 0.25 (n=10, not_eval=0) | 0.2 (n=10, not_eval=0) | 0.05 (n=10, not_eval=0) | 0.25 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) |
| Priority Focus Correctness | 0/1, rubric matching urgency/steady language to risk level | 0.0 (n=10, not_eval=0) | 0.6 (n=10, not_eval=0) | 0.5 (n=10, not_eval=0) | 0.4 (n=10, not_eval=0) | 0.5 (n=10, not_eval=0) | 0.6 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) |

### 6b.1c Grounding Recall detail (deterministic sub-scores by evidence source)

Grounding Recall (6b.1) is the mean of whichever of these three are available per record.

| Metric | Scale | Full System | Without Retrieval | Without Persona | Without SHAP | Without Cognitive | Risk Only | LLM Only |
|---|---|---|---|---|---|---|---|---|
| SHAP grounding | 0-1, deterministic recall | Not Evaluated (n_not_evaluated=10) | 0.0063 (n=10, not_eval=0) | 0.0 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) | 0.016 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) | Not Evaluated (n_not_evaluated=10) |
| Risk-reasons grounding | 0-1, deterministic recall | Not Evaluated (n_not_evaluated=10) | 0.0216 (n=10, not_eval=0) | 0.0091 (n=10, not_eval=0) | 0.0091 (n=10, not_eval=0) | 0.0216 (n=10, not_eval=0) | 0.0091 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) |
| Cognitive-state grounding | 0-1, deterministic recall | Not Evaluated (n_not_evaluated=10) | 0.0 (n=10, not_eval=0) | 0.0261 (n=10, not_eval=0) | 0.01 (n=10, not_eval=0) | Not Evaluated (n_not_evaluated=10) | Not Evaluated (n_not_evaluated=10) | Not Evaluated (n_not_evaluated=10) |

### 6b.2 Population-level metrics by configuration (deterministic)

| Configuration | Recommendation Diversity (unique ratio) | Mean pairwise dissimilarity | Tone differentiation (within - across persona similarity) |
|---|---|---|---|
| Full System | 0.3 (n=10) | 0.355 | -0.053 (within=0.6, across=0.653) |
| Without Retrieval | 0.9 (n=10) | 0.893 | 0.011 (within=0.112, across=0.101) |
| Without Persona | 1.0 (n=10) | 0.902 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| Without SHAP | 0.9 (n=10) | 0.883 | 0.048 (within=0.113, across=0.065) |
| Without Cognitive | 0.9 (n=10) | 0.911 | -0.005 (within=0.103, across=0.108) |
| Risk Only | 1.0 (n=10) | 0.892 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |
| LLM Only | 1.0 (n=10) | 0.958 | Not Evaluated (within=Not Evaluated, across=Not Evaluated) |

### 6b.3 Statistical detail (mean, std, 95% CI; observed only)

_95% CIs use a normal approximation and are noted as approximate for n < 30. Cells with fewer than 2 observations report `Not Evaluated` for std/CI._

| Metric | Configuration | n | Mean | Std | 95% CI |
|---|---|---|---|---|---|
| Educational Usefulness | Without Persona | 10 | 0.0849 | 0.1181 | [0.0117, 0.1581] |
| Educational Usefulness | Without SHAP | 10 | 0.0397 | 0.0926 | [-0.0177, 0.0971] |
| Educational Usefulness | Without Cognitive | 10 | 0.0019 | 0.006 | [-0.0018, 0.0056] |
| Educational Usefulness | Risk Only | 10 | 0.0679 | 0.0946 | [0.0092, 0.1266] |
| Persona Consistency | Full System | 10 | 0.6 | 0.0 | [0.6, 0.6] |
| Persona Consistency | Without Retrieval | 10 | 0.1115 | 0.029 | [0.0935, 0.1295] |
| Persona Consistency | Without SHAP | 10 | 0.1125 | 0.0111 | [0.1056, 0.1194] |
| Persona Consistency | Without Cognitive | 10 | 0.103 | 0.0253 | [0.0873, 0.1187] |
| Grounding Recall | Without Retrieval | 10 | 0.0093 | 0.0198 | [-0.003, 0.0216] |
| Grounding Recall | Without Persona | 10 | 0.0117 | 0.0205 | [-0.001, 0.0244] |
| Grounding Recall | Without SHAP | 10 | 0.0095 | 0.0162 | [-0.0006, 0.0196] |
| Grounding Recall | Without Cognitive | 10 | 0.0188 | 0.0403 | [-0.0062, 0.0438] |
| Grounding Recall | Risk Only | 10 | 0.0091 | 0.0288 | [-0.0087, 0.0269] |
| BLEU Score | Full System | 10 | 0.0004 | 0.0008 | [-0.0001, 0.0009] |
| BLEU Score | Without Retrieval | 10 | 0.0093 | 0.0181 | [-0.0018, 0.0205] |
| BLEU Score | Without Persona | 10 | 0.0049 | 0.0023 | [0.0034, 0.0063] |
| BLEU Score | Without SHAP | 10 | 0.0042 | 0.0019 | [0.0031, 0.0054] |
| BLEU Score | Without Cognitive | 10 | 0.0043 | 0.0028 | [0.0025, 0.006] |
| BLEU Score | Risk Only | 10 | 0.0073 | 0.0049 | [0.0043, 0.0104] |
| BLEU Score | LLM Only | 10 | 0.0051 | 0.0022 | [0.0037, 0.0064] |
| Semantic Similarity | Full System | 10 | 0.0925 | 0.1048 | [0.0275, 0.1575] |
| Semantic Similarity | Without Retrieval | 10 | 0.4453 | 0.1624 | [0.3447, 0.546] |
| Semantic Similarity | Without Persona | 10 | 0.4178 | 0.189 | [0.3007, 0.535] |
| Semantic Similarity | Without SHAP | 10 | 0.4197 | 0.1526 | [0.3251, 0.5143] |
| Semantic Similarity | Without Cognitive | 10 | 0.4338 | 0.1526 | [0.3392, 0.5283] |
| Semantic Similarity | Risk Only | 10 | 0.4465 | 0.1511 | [0.3529, 0.5401] |
| Semantic Similarity | LLM Only | 10 | 0.548 | 0.1475 | [0.4565, 0.6394] |
| Recommendation Quality | Full System | 10 | 1.9 | 0.3162 | [1.704, 2.096] |
| Recommendation Quality | Without Retrieval | 10 | 1.9 | 0.8756 | [1.3573, 2.4427] |
| Recommendation Quality | Without Persona | 10 | 2.9 | 0.3162 | [2.704, 3.096] |
| Recommendation Quality | Without SHAP | 10 | 1.7 | 0.6749 | [1.2817, 2.1183] |
| Recommendation Quality | Without Cognitive | 10 | 2.0 | 0.8165 | [1.4939, 2.5061] |
| Recommendation Quality | Risk Only | 10 | 2.4 | 0.6992 | [1.9666, 2.8334] |
| Recommendation Quality | LLM Only | 10 | 1.9 | 0.3162 | [1.704, 2.096] |
| Evidence Fusion Correctness | Full System | 10 | 0.0 | 0.0 | [0.0, 0.0] |
| Evidence Fusion Correctness | Without Retrieval | 10 | 0.2 | 0.1972 | [0.0778, 0.3222] |
| Evidence Fusion Correctness | Without Persona | 10 | 0.25 | 0.1667 | [0.1467, 0.3533] |
| Evidence Fusion Correctness | Without SHAP | 10 | 0.2 | 0.2582 | [0.04, 0.36] |
| Evidence Fusion Correctness | Without Cognitive | 10 | 0.05 | 0.1054 | [-0.0153, 0.1153] |
| Evidence Fusion Correctness | Risk Only | 10 | 0.25 | 0.2635 | [0.0867, 0.4133] |
| Priority Focus Correctness | Full System | 10 | 0.0 | 0.0 | [0.0, 0.0] |
| Priority Focus Correctness | Without Retrieval | 10 | 0.6 | 0.5164 | [0.2799, 0.9201] |
| Priority Focus Correctness | Without Persona | 10 | 0.5 | 0.527 | [0.1733, 0.8267] |
| Priority Focus Correctness | Without SHAP | 10 | 0.4 | 0.5164 | [0.0799, 0.7201] |
| Priority Focus Correctness | Without Cognitive | 10 | 0.5 | 0.527 | [0.1733, 0.8267] |
| Priority Focus Correctness | Risk Only | 10 | 0.6 | 0.5164 | [0.2799, 0.9201] |
| SHAP grounding | Without Retrieval | 10 | 0.0063 | 0.0133 | [-0.0019, 0.0145] |
| SHAP grounding | Without Persona | 10 | 0.0 | 0.0 | [0.0, 0.0] |
| SHAP grounding | Without Cognitive | 10 | 0.016 | 0.0409 | [-0.0093, 0.0413] |
| Risk-reasons grounding | Without Retrieval | 10 | 0.0216 | 0.0462 | [-0.0071, 0.0503] |
| Risk-reasons grounding | Without Persona | 10 | 0.0091 | 0.0288 | [-0.0087, 0.0269] |
| Risk-reasons grounding | Without SHAP | 10 | 0.0091 | 0.0288 | [-0.0087, 0.0269] |
| Risk-reasons grounding | Without Cognitive | 10 | 0.0216 | 0.0462 | [-0.0071, 0.0503] |
| Risk-reasons grounding | Risk Only | 10 | 0.0091 | 0.0288 | [-0.0087, 0.0269] |
| Cognitive-state grounding | Without Retrieval | 10 | 0.0 | 0.0 | [0.0, 0.0] |
| Cognitive-state grounding | Without Persona | 10 | 0.0261 | 0.042 | [0.0001, 0.0521] |
| Cognitive-state grounding | Without SHAP | 10 | 0.01 | 0.0211 | [-0.0031, 0.0231] |

### 6b.4 Paired comparison vs. Full System (same user+question, evidence config differs)

Pairs are formed only where BOTH the Full System record and the ablation-arm record for the same `(user_id, question)` have a real numeric score for that metric -- no imputation. `p_value`/`t_stat` (paired t-test) require `scipy`; if unavailable, only the mean/std of paired differences is reported and `p_value` reads `Not Evaluated`. The `Recommended test` columns additionally choose between a paired t-test and a Wilcoxon signed-rank test per-comparison, based on a Shapiro-Wilk normality check of the paired differences (Wilcoxon is preferred whenever normality is rejected or cannot be checked, per the study's stated test preference), and report an effect size alongside it.

| Configuration | Metric | n pairs | Mean diff (arm - Full System) | Std diff | t-stat | p-value | Recommended test | Recommended p-value | Effect size |
|---|---|---|---|---|---|---|---|---|---|
| Without Retrieval | Persona Consistency | 10 | -0.4885 | 0.029 | -53.2909 | 0.0 | Wilcoxon signed-rank | 0.002 | -0.9794 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Retrieval | BLEU Score | 10 | 0.0089 | 0.0183 | 1.5496 | 0.1557 | Wilcoxon signed-rank | 0.0098 | 0.8171 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Retrieval | Semantic Similarity | 10 | 0.3528 | 0.2076 | 5.376 | 0.0004 | paired t-test | 0.0004 | 1.7 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Retrieval | Recommendation Quality | 10 | 0.0 | 1.0541 | 0.0 | 1.0 | paired t-test | 1.0 | 0.0 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Retrieval | Evidence Fusion Correctness | 10 | 0.2 | 0.1972 | 3.2071 | 0.0107 | Wilcoxon signed-rank | 0.0312 | 0.8793 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Retrieval | Priority Focus Correctness | 10 | 0.6 | 0.5164 | 3.6742 | 0.0051 | Wilcoxon signed-rank | 0.0312 | 0.8793 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Persona | BLEU Score | 10 | 0.0045 | 0.0029 | 4.9455 | 0.0008 | paired t-test | 0.0008 | 1.5639 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Persona | Semantic Similarity | 10 | 0.3254 | 0.2238 | 4.5984 | 0.0013 | paired t-test | 0.0013 | 1.4541 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Persona | Recommendation Quality | 10 | 1.0 | 0.4714 | 6.7082 | 0.0001 | Wilcoxon signed-rank | 0.0039 | 0.9619 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Persona | Evidence Fusion Correctness | 10 | 0.25 | 0.1667 | 4.7434 | 0.0011 | Wilcoxon signed-rank | 0.0078 | 0.9405 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Persona | Priority Focus Correctness | 10 | 0.5 | 0.527 | 3.0 | 0.015 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Persona Consistency | 10 | -0.4875 | 0.0111 | -139.2857 | 0.0 | Wilcoxon signed-rank | 0.002 | -0.9794 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | BLEU Score | 10 | 0.0039 | 0.0024 | 5.0986 | 0.0006 | paired t-test | 0.0006 | 1.6123 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without SHAP | Semantic Similarity | 10 | 0.3272 | 0.2021 | 5.1194 | 0.0006 | paired t-test | 0.0006 | 1.6189 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without SHAP | Recommendation Quality | 10 | -0.2 | 0.6325 | -1.0 | 0.3434 | Wilcoxon signed-rank | 0.625 | -0.2444 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Evidence Fusion Correctness | 10 | 0.2 | 0.2582 | 2.4495 | 0.0368 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without SHAP | Priority Focus Correctness | 10 | 0.4 | 0.5164 | 2.4495 | 0.0368 | Wilcoxon signed-rank | 0.125 | 0.7671 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Persona Consistency | 10 | -0.497 | 0.0253 | -62.125 | 0.0 | Wilcoxon signed-rank | 0.002 | -0.9794 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | BLEU Score | 10 | 0.0039 | 0.0034 | 3.651 | 0.0053 | paired t-test | 0.0053 | 1.1545 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Cognitive | Semantic Similarity | 10 | 0.3413 | 0.1931 | 5.5873 | 0.0003 | paired t-test | 0.0003 | 1.7669 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Without Cognitive | Recommendation Quality | 10 | 0.1 | 0.8756 | 0.3612 | 0.7263 | Wilcoxon signed-rank | 1.0 | 0.0 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Evidence Fusion Correctness | 10 | 0.05 | 0.1054 | 1.5 | 0.1679 | Wilcoxon signed-rank | 0.5 | 0.4769 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Without Cognitive | Priority Focus Correctness | 10 | 0.5 | 0.527 | 3.0 | 0.015 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Risk Only | BLEU Score | 10 | 0.0069 | 0.0052 | 4.2355 | 0.0022 | paired t-test | 0.0022 | 1.3394 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| Risk Only | Semantic Similarity | 10 | 0.354 | 0.2084 | 5.3713 | 0.0004 | Wilcoxon signed-rank | 0.0039 | 0.9125 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Risk Only | Recommendation Quality | 10 | 0.5 | 0.7071 | 2.2361 | 0.0522 | Wilcoxon signed-rank | 0.125 | 0.5798 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Risk Only | Evidence Fusion Correctness | 10 | 0.25 | 0.2635 | 3.0 | 0.015 | Wilcoxon signed-rank | 0.0625 | 0.833 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| Risk Only | Priority Focus Correctness | 10 | 0.6 | 0.5164 | 3.6742 | 0.0051 | Wilcoxon signed-rank | 0.0312 | 0.8793 (matched-pairs rank-biserial r (approximate, from normal-approx Z)) |
| LLM Only | BLEU Score | 10 | 0.0047 | 0.0027 | 5.4765 | 0.0004 | paired t-test | 0.0004 | 1.7318 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| LLM Only | Semantic Similarity | 10 | 0.4555 | 0.2267 | 6.3537 | 0.0001 | paired t-test | 0.0001 | 2.0092 (Cohen's d_z (mean paired diff / std of paired diffs)) |
| LLM Only | Recommendation Quality | 10 | 0.0 | 0.0 | nan | nan | Wilcoxon signed-rank | Not Evaluated | Not Evaluated |

### 6b.5 Discussion (automatic interpretation of the paired comparisons above)

Interpretive sentences below are generated directly from the `mean_diff` values in 6b.4 -- no explanation is invented beyond what those measured differences show. Percent change is relative to the Full System mean for that metric where that mean is a real (non-zero) observed value; otherwise only the raw mean difference is stated.

- Under **Without Retrieval**, Persona Consistency decreased by **81.4%** relative to Full System (mean diff = -0.4885), a statistically significant difference (p = 0.002), suggesting the evidence removed in this arm meaningfully contributes to persona consistency when present in the Full System.
- Under **Without Retrieval**, BLEU Score increased by **2225.0%** relative to Full System (mean diff = 0.0089), a statistically significant difference (p = 0.0098), suggesting the evidence removed in this arm meaningfully detracts from bleu score when present in the Full System.
- Under **Without Retrieval**, Semantic Similarity increased by **381.4%** relative to Full System (mean diff = 0.3528), a statistically significant difference (p = 0.0004), suggesting the evidence removed in this arm meaningfully detracts from semantic similarity when present in the Full System.
- Under **Without Retrieval**, Evidence Fusion Correctness increased by a mean of **0.2** relative to Full System, a statistically significant difference (p = 0.0312) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without Retrieval**, Priority Focus Correctness increased by a mean of **0.6** relative to Full System, a statistically significant difference (p = 0.0312) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without Persona**, BLEU Score increased by **1125.0%** relative to Full System (mean diff = 0.0045), a statistically significant difference (p = 0.0008), suggesting the evidence removed in this arm meaningfully detracts from bleu score when present in the Full System.
- Under **Without Persona**, Semantic Similarity increased by **351.8%** relative to Full System (mean diff = 0.3254), a statistically significant difference (p = 0.0013), suggesting the evidence removed in this arm meaningfully detracts from semantic similarity when present in the Full System.
- Under **Without Persona**, Recommendation Quality increased by **52.6%** relative to Full System (mean diff = 1.0), a statistically significant difference (p = 0.0039), suggesting the evidence removed in this arm meaningfully detracts from recommendation quality when present in the Full System.
- Under **Without Persona**, Evidence Fusion Correctness increased by a mean of **0.25** relative to Full System, a statistically significant difference (p = 0.0078) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without Persona**, Priority Focus Correctness increased by a mean of **0.5** relative to Full System (not statistically significant at p = 0.0625) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without SHAP**, Persona Consistency decreased by **81.2%** relative to Full System (mean diff = -0.4875), a statistically significant difference (p = 0.002), suggesting the evidence removed in this arm meaningfully contributes to persona consistency when present in the Full System.
- Under **Without SHAP**, BLEU Score increased by **975.0%** relative to Full System (mean diff = 0.0039), a statistically significant difference (p = 0.0006), suggesting the evidence removed in this arm meaningfully detracts from bleu score when present in the Full System.
- Under **Without SHAP**, Semantic Similarity increased by **353.7%** relative to Full System (mean diff = 0.3272), a statistically significant difference (p = 0.0006), suggesting the evidence removed in this arm meaningfully detracts from semantic similarity when present in the Full System.
- Under **Without SHAP**, Recommendation Quality decreased by **10.5%** relative to Full System (mean diff = -0.2) (not statistically significant at p = 0.625), suggesting the evidence removed in this arm meaningfully contributes to recommendation quality when present in the Full System.
- Under **Without SHAP**, Evidence Fusion Correctness increased by a mean of **0.2** relative to Full System (not statistically significant at p = 0.0625) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without SHAP**, Priority Focus Correctness increased by a mean of **0.4** relative to Full System (not statistically significant at p = 0.125) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without Cognitive**, Persona Consistency decreased by **82.8%** relative to Full System (mean diff = -0.497), a statistically significant difference (p = 0.002), suggesting the evidence removed in this arm meaningfully contributes to persona consistency when present in the Full System.
- Under **Without Cognitive**, BLEU Score increased by **975.0%** relative to Full System (mean diff = 0.0039), a statistically significant difference (p = 0.0053), suggesting the evidence removed in this arm meaningfully detracts from bleu score when present in the Full System.
- Under **Without Cognitive**, Semantic Similarity increased by **369.0%** relative to Full System (mean diff = 0.3413), a statistically significant difference (p = 0.0003), suggesting the evidence removed in this arm meaningfully detracts from semantic similarity when present in the Full System.
- Under **Without Cognitive**, Recommendation Quality increased by **5.3%** relative to Full System (mean diff = 0.1) (not statistically significant at p = 1.0), suggesting the evidence removed in this arm meaningfully detracts from recommendation quality when present in the Full System.
- Under **Without Cognitive**, Evidence Fusion Correctness increased by a mean of **0.05** relative to Full System (not statistically significant at p = 0.5) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Without Cognitive**, Priority Focus Correctness increased by a mean of **0.5** relative to Full System (not statistically significant at p = 0.0625) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Risk Only**, BLEU Score increased by **1725.0%** relative to Full System (mean diff = 0.0069), a statistically significant difference (p = 0.0022), suggesting the evidence removed in this arm meaningfully detracts from bleu score when present in the Full System.
- Under **Risk Only**, Semantic Similarity increased by **382.7%** relative to Full System (mean diff = 0.354), a statistically significant difference (p = 0.0039), suggesting the evidence removed in this arm meaningfully detracts from semantic similarity when present in the Full System.
- Under **Risk Only**, Recommendation Quality increased by **26.3%** relative to Full System (mean diff = 0.5) (not statistically significant at p = 0.125), suggesting the evidence removed in this arm meaningfully detracts from recommendation quality when present in the Full System.
- Under **Risk Only**, Evidence Fusion Correctness increased by a mean of **0.25** relative to Full System (not statistically significant at p = 0.0625) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **Risk Only**, Priority Focus Correctness increased by a mean of **0.6** relative to Full System, a statistically significant difference (p = 0.0312) (Full System baseline mean not available/zero, so a percentage change could not be computed).
- Under **LLM Only**, BLEU Score increased by **1175.0%** relative to Full System (mean diff = 0.0047), a statistically significant difference (p = 0.0004), suggesting the evidence removed in this arm meaningfully detracts from bleu score when present in the Full System.
- Under **LLM Only**, Semantic Similarity increased by **492.4%** relative to Full System (mean diff = 0.4555), a statistically significant difference (p = 0.0001), suggesting the evidence removed in this arm meaningfully detracts from semantic similarity when present in the Full System.

- Under **Without Retrieval**, no measured difference vs Full System: Recommendation Quality.
- Under **LLM Only**, no measured difference vs Full System: Recommendation Quality.

## 6e. Correlation Analysis

Spearman rank correlation between the confidence level `confidence.assess_confidence()` actually assigned (mapped Low=1, Medium=2, High=3 -- the same ordering already encoded in that module's own rubric) and each quality metric, computed only over records with a real confidence level and a real numeric metric value. Reported `Not Evaluated` when fewer than 3 such paired observations exist, or when `scipy` is unavailable.

| Comparison | n | Spearman r | p-value |
|---|---|---|---|
| Confidence vs Recommendation Quality | 60 | -0.4018 | 0.0015 |
| Confidence vs Educational Usefulness | 40 | -0.4179 | 0.0073 |
| Confidence vs Grounding Recall | 50 | 0.015 | 0.9179 |

## 6f. Error Analysis

Computed strictly from the observed records above -- failure counts, parser fallback counts, response lengths, and confidence labels already captured on each `EvalRecord`.

- Configuration with the highest failure rate: **Full System** (failure rate = 0.0)
- Overall parser fallback frequency: **0.0**
- Overall structured-response compliance rate: **1.0**
- Overall confidence distribution: {'High': 30, 'Low': 20, 'Medium': 10}

| Configuration | n | Failures | Failure rate | Parser fallback freq. | Structured compliance rate | Avg. response length (chars) | Confidence distribution |
|---|---|---|---|---|---|---|---|
| Full System | 10 | 0 | 0.0 | Not Evaluated | Not Evaluated | 92.0 | {'High': 10} |
| Without Retrieval | 10 | 0 | 0.0 | 0.0 | 1.0 | 234.5 | {'High': 10} |
| Without Persona | 10 | 0 | 0.0 | 0.0 | 1.0 | 335.1 | {'Low': 10} |
| Without SHAP | 10 | 0 | 0.0 | 0.0 | 1.0 | 256.8 | {'Medium': 10} |
| Without Cognitive | 10 | 0 | 0.0 | 0.0 | 1.0 | 274.5 | {'High': 10} |
| Risk Only | 10 | 0 | 0.0 | 0.0 | 1.0 | 316.1 | {'Low': 10} |
| LLM Only | 10 | 0 | 0.0 | 0.0 | 1.0 | 313.5 | Not Evaluated |

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
- Results are based on the specific evaluated user population (2 user(s)) and question set (5 question(s) -- §1.3) in this run; they may not generalize to other students, personas, risk profiles, or question phrasings not represented here.
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


## 6h. BLEU Score and Semantic Similarity (extended statistics)

Computed per generated `mentor_response_text` against a single expert gold reference answer for the same question (see `outputs/gold_references.json`, auto-generated once from the built-in default questions and never overwritten thereafter). BLEU uses `nltk.translate.bleu_score.sentence_bleu` with `SmoothingFunction().method1`; Semantic Similarity is cosine similarity between `sentence-transformers` `all-MiniLM-L6-v2` embeddings of the reference and the generated response. Both report `Not Evaluated` (never a fabricated number) when a question has no gold reference, the record has no captured response text, or the optional `nltk` / `sentence-transformers` dependency is not installed -- see §9 for exact package names to add.

### 6h.1 Overall (all configurations combined)

| Metric | n | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| BLEU Score | 70 | 0.0051 | 0.004 | 0.0074 | 0.0 | 0.0603 |
| Semantic Similarity | 70 | 0.4005 | 0.4315 | 0.198 | 0.0096 | 0.7369 |

### 6h.2 By configuration

| Configuration | BLEU n | BLEU Mean | BLEU Median | BLEU Std | BLEU Min | BLEU Max | Semantic Sim. n | Semantic Sim. Mean | Semantic Sim. Median | Semantic Sim. Std | Semantic Sim. Min | Semantic Sim. Max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full System | 10 | 0.0004 | 0.0 | 0.0008 | 0.0 | 0.002 | 10 | 0.0925 | 0.0394 | 0.1048 | 0.0096 | 0.3398 |
| Without Retrieval | 10 | 0.0093 | 0.0039 | 0.0181 | 0.0011 | 0.0603 | 10 | 0.4453 | 0.442 | 0.1624 | 0.2221 | 0.6505 |
| Without Persona | 10 | 0.0049 | 0.0042 | 0.0023 | 0.0011 | 0.008 | 10 | 0.4178 | 0.4149 | 0.189 | 0.0976 | 0.7061 |
| Without SHAP | 10 | 0.0042 | 0.004 | 0.0019 | 0.0011 | 0.0081 | 10 | 0.4197 | 0.4579 | 0.1526 | 0.0658 | 0.6212 |
| Without Cognitive | 10 | 0.0043 | 0.004 | 0.0028 | 0.0011 | 0.0112 | 10 | 0.4338 | 0.4153 | 0.1526 | 0.2185 | 0.6555 |
| Risk Only | 10 | 0.0073 | 0.0063 | 0.0049 | 0.0026 | 0.0191 | 10 | 0.4465 | 0.4584 | 0.1511 | 0.2028 | 0.6571 |
| LLM Only | 10 | 0.0051 | 0.0048 | 0.0022 | 0.0014 | 0.0082 | 10 | 0.548 | 0.5693 | 0.1475 | 0.3104 | 0.7369 |

Per-record BLEU and semantic-similarity values (one row per (user, question, config)) are also written to `outputs/bleu_scores.csv` and `outputs/semantic_similarity.csv` respectively.


## 6i. LLM Only Baseline

`LLM Only` is a new ablation arm that disables retrieval, persona, risk scoring, SHAP, and cognitive-state evidence entirely, and answers the student's raw question using only a generic mentor system prompt (see `mask_context()` / `run_ablation_call()` in `rag_ablation.py`, §3). It serves as a lower-bound baseline: any configuration that includes real evidence should be expected to outperform it on evidence-grounded metrics (Evidence Fusion, Grounding Recall, Educational Usefulness) if that evidence is actually being used by the pipeline.

| Metric | LLM Only (mean) | Full System (mean) | Difference (Full System - LLM Only) |
|---|---|---|---|
| Educational Usefulness | Not Evaluated | Not Evaluated | Not Evaluated |
| Persona Consistency | Not Evaluated | 0.6 | Not Evaluated |
| Grounding Recall | Not Evaluated | Not Evaluated | Not Evaluated |
| BLEU Score | 0.0051 | 0.0004 | -0.0047 |
| Semantic Similarity | 0.548 | 0.0925 | -0.4555 |
| Recommendation Quality | 1.9 | 1.9 | 0.0 |
| Evidence Fusion Correctness | Not Evaluated | 0.0 | Not Evaluated |
| Priority Focus Correctness | Not Evaluated | 0.0 | Not Evaluated |
| SHAP grounding | Not Evaluated | Not Evaluated | Not Evaluated |
| Risk-reasons grounding | Not Evaluated | Not Evaluated | Not Evaluated |
| Cognitive-state grounding | Not Evaluated | Not Evaluated | Not Evaluated |

A positive difference above means Full System outperformed the LLM Only baseline on that metric in this run; `Not Evaluated` means at least one side lacked a real observed mean for that metric.


## 7a. Reproducibility Summary

| Item | Value |
|---|---|
| Backend SHA256 (hash of imported backend module source) | `842e41aaa533e18e72d08824cd53543efdee9f96756b2912230c0e48516a90ae` |
| Ollama model | `qwen2.5:3b` |
| Live LLM active | True |
| Number of evaluated users | 2 |
| Number of evaluated questions | 5 |
| Number of ablation configurations | 7 |
| Total evaluations (users x questions x configs) | 70 |
| Timestamp (UTC) | 2026-07-06T11:22:14.672838+00:00 |

## 7. Reproducing This Run

```
cd <backend project root>   # where the `services` package lives
export EVAL_USER_IDS="<real_user_id_1>,<real_user_id_2>"
export USE_LIVE_LLM=true
python ml/rag_ablation/rag_ablation.py
```
Every value in this report is derived from `outputs/records.jsonl`, produced by this exact run; re-running with the same user IDs, questions, backend code, and a live Ollama instance should reproduce the same real-data metrics up to LLM sampling variance (temperature=0.5, top_p=0.9, as configured in the real `mentor_service.ask_mentor`).