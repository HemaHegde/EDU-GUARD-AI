# EduGuard-AI — Psychologically Grounded, Explainable Student Disengagement Prediction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Access](https://img.shields.io/badge/paper-IEEE_Access-green.svg)](https://ieeeaccess.ieee.org/)

EduGuard-AI is an open-source, IEEE Access research system that combines:
- **XGBoost behavioural risk prediction** on the Open University Learning Analytics Dataset (OULAD)
- **SHAP explainability** for per-student feature attribution
- **K-Means persona clustering** (6 learner archetypes)
- **GRU-based cognitive state modelling** from learning-session time series
- **RAG-powered AI Mentor** (FAISS + Ollama/qwen2.5:3b) with hallucination safety

---

## Quick-Start

### 1. Clone and install

```bash
git clone <this-repo>
cd edu-ai
pip install -r requirements.txt
```

### 2. Verify Ollama is running (for AI Mentor / RAG ablation)

```bash
ollama serve          # in one terminal
ollama pull qwen2.5:3b
```

### 3. Run the full IEEE evaluation pipeline

```bash
cd ml
python run_all_ieee_experiments.py
```

This runs Sections C through L in order and saves all outputs to `ml/ieee_results/`.

### 4. Run individual experiments

| Script | Purpose |
|---|---|
| `ml/run_ieee_evaluation.py` | Core XGBoost + SHAP + clustering evaluation |
| `ml/section_c_extended_baselines.py` | LR / RF / SVM / LightGBM / CatBoost baselines |
| `ml/section_d_kfold_validation.py` | 5-fold cross-validation |
| `ml/section_e_statistical_tests.py` | DeLong AUC test + McNemar |
| `ml/section_f_shap_explainability.py` | SHAP dependence + waterfall plots |
| `ml/section_g_clustering_validation.py` | Cluster stability: 20 seeds, t-SNE, UMAP |
| `ml/section_h_cognitive_validation.py` | LSTM / GRU / CNN / Ensemble cognitive model |
| `ml/section_i_ablation_study.py` | 5-config feature ablation study |
| `ml/section_jkl_fairness_robustness_deployment.py` | Fairness (J) + Robustness (K) + Latency (L) |
| `ml/rag_ablation/rag_ablation.py` | 7-config RAG ablation (requires Ollama + Supabase) |
| `ml/ethics_fairness/ethics_fairness.py` | Post-hoc fairness audit (6 demographics) |
| `ml/error_analysis/error_analysis.py` | FP/FN/hardest-case extraction |
| `ml/ai_mentor_failure_analysis/ai_mentor_failure_analysis.py` | 8-scenario robustness test |
| `ml/reproducibility/reproducibility.py` | Environment + artifact + seed documentation |
| `ml/external_validation/preprocess_assistments.py` | ASSISTments preprocessing |
| `ml/external_validation/generate_external_predictions.py` | External validation on ASSISTments |

---

## Project Structure

```
edu-ai/
├── README.md                         ← this file
├── config.yaml                       ← central configuration (seeds, thresholds, paths)
├── requirements.txt                  ← pinned Python dependencies
│
├── backend/
│   ├── app/                          ← FastAPI application
│   ├── routes/                       ← 10 REST API route modules
│   └── services/                     ← core modular pipeline
│       ├── mentor_service.py         ← AI Mentor orchestrator (ask_mentor)
│       ├── context_builder.py        ← student context assembly
│       ├── prompt_builder.py         ← modular prompt construction (8 sections)
│       ├── retrieval.py              ← FAISS RAG + similarity threshold
│       ├── confidence.py             ← dual-confidence rubric + hallucination safety
│       ├── reasoning_layer.py        ← multi-signal evidence reasoning
│       ├── risk_service.py           ← XGBoost risk prediction wrapper
│       ├── shap_service.py           ← SHAP feature attribution
│       ├── persona_service.py        ← K-Means persona assignment
│       └── cognitive_adapter.py     ← GRU cognitive state adapter
│
├── ml/
│   ├── final_student_psychology_dataset.csv  ← OULAD features (32,593 students)
│   ├── academic_risk_xgboost.pkl             ← frozen XGBoost model
│   ├── persona_kmeans.pkl                    ← frozen K-Means (6 clusters)
│   ├── persona_scaler.pkl                    ← frozen StandardScaler
│   │
│   ├── ieee_results/                 ← all IEEE section outputs (35 figures, 28 CSVs)
│   ├── rag_ablation/                 ← RAG ablation (7 configs, BLEU, sem-sim, 9 figures)
│   ├── ethics_fairness/              ← fairness audit (6 demographics, 4 figures)
│   ├── error_analysis/               ← FP/FN/hardest-cases (3 figures, 6 CSVs)
│   ├── ai_mentor_failure_analysis/  ← 8 adversarial scenarios (2 figures)
│   ├── external_validation/          ← ASSISTments pipeline (23 figures, 7 reports)
│   ├── reproducibility/              ← environment/artifact snapshot
│   └── video_intelligence/           ← GRU/LSTM/CNN cognitive models
│
├── frontend/                         ← React web interface
│
├── datasets/
│   └── ASSISTments/                  ← external validation dataset
│
└── studentAssessment.csv             ← OULAD raw assessment data
    studentVle.csv                    ← OULAD raw VLE interaction data
    studentRegistration.csv           ← OULAD raw registration data
```

---

## Dataset

**Primary dataset:** Open University Learning Analytics Dataset (OULAD)
- 32,593 student–module enrolment records
- 7 engineered behavioural features (interaction clicks, assessment scores, active days, inactivity, engagement trend, assessment consistency, engagement variability)
- Binary label: `psychological_risk` (1 = Withdrawn/Fail, 0 = otherwise)

**External validation dataset:** ASSISTments 2012–2013
- 63,174 student–class observation windows
- 6.1 million problem-interaction logs
- Preprocessed to the same 7 behavioural features by `preprocess_assistments.py`

---

## Key Results

| Metric | XGBoost | Random Forest | Logistic Regression |
|---|---|---|---|
| Accuracy | **0.8891** | 0.8865 | 0.8770 |
| Precision | 0.9474 | 0.9506 | 0.9348 |
| Recall | 0.8364 | 0.8280 | 0.8245 |
| F1-Score | **0.8884** | 0.8851 | 0.8762 |
| ROC-AUC | **0.9486** | 0.9468 | 0.9372 |

**5-Fold Cross-Validation (XGBoost):** Accuracy = 0.8857 ± 0.003, ROC-AUC = 0.9474 ± 0.003

**Fairness:** 8 four-fifths-rule flags across 6 demographics (largest effect: highest_education, Cohen's h = 0.479)

**External Validation (ASSISTments):** ROC-AUC = 0.8556 on 63,174 records; SHAP rank correlation ρ = 0.786 (p = 0.036)

**AI Mentor (RAG Ablation):** 7/7 configs, 0 failures, Persona Consistency = 0.60 (Full System vs. 0.11 without retrieval)

---

## Reproducibility

All results use `random_state=42` throughout. Model SHA-256 hashes are documented in `ml/reproducibility/reports/reproducibility_report.md`.

To regenerate the reproducibility snapshot:
```bash
cd ml/reproducibility
python reproducibility.py
```

---

## Configuration

All tunable constants (thresholds, seeds, batch sizes, model paths) are in [`config.yaml`](config.yaml).

---

## Citation

If you use EduGuard-AI in your research, please cite:

```
[IEEE Access paper citation — to be added on acceptance]
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
