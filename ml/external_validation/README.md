# EduGuard-AI — External Validation Pipeline

`ml/external_validation/`

---

## 1. Purpose

This folder contains the complete **external validation pipeline** for
**EduGuard-AI**, the psychologically grounded, explainable disengagement
prediction framework described in the accompanying IEEE Access paper.

The purpose of this pipeline is to assess whether the behavioural risk
model trained on the **Open University Learning Analytics Dataset
(OULAD)** generalises to an independent, structurally different online
learning platform. External validation of this kind is essential for
demonstrating that the model's predictive power reflects genuine,
transferable behavioural patterns of student disengagement, rather than
dataset-specific artefacts of OULAD.

The target validation dataset for this pipeline is **ASSISTments
2012–2013**, a large-scale, publicly available intelligent tutoring
system log.

---

## 2. Dataset

**ASSISTments 2012–2013**

ASSISTments is a web-based intelligent tutoring platform used in K–12
mathematics education. The 2012–2013 release provides fine-grained,
timestamped problem-log interaction data suitable for behavioural
feature reconstruction.

| Property | Value |
|---|---|
| Total interaction logs | 6,123,270 |
| Unique students | 46,347 |
| Granularity | Individual problem-log interactions |
| Key raw fields | `user_id`, `problem_log_id`, `problem_id`, `assignment_id`, `sequence_id`, `start_time`, `end_time`, `correct`, `attempt_count`, `hint_count`, `actions`, `ms_first_response` |

Because ASSISTments does not natively expose the seven behavioural
constructs used by EduGuard-AI's OULAD-trained model, this pipeline
reconstructs each construct from the raw interaction logs through a
documented, auditable feature-engineering process, producing a
student-level behavioural profile suitable for use with the existing
model schema.

---

## 3. Completed Components

The following stages of the external validation pipeline have been
completed:

- ✔ **Dataset audit** — verification of the raw ASSISTments schema,
  column availability, and data quality characteristics prior to
  processing.
- ✔ **Feature mapping** — a reviewed and approved mapping from raw
  ASSISTments interaction fields to the seven fixed OULAD behavioural
  features required by the EduGuard-AI model.
- ✔ **Preprocessing** — timestamp parsing and validation, duplicate
  removal, duration computation, and removal of invalid/outlier
  durations, followed by grouped temporal feature engineering.
- ✔ **`assistments_student_profiles.csv` generation** — production and
  verification of the final, model-ready student–class behavioural
  feature table.

---

## 4. Generated Behavioural Features

The pipeline computes exactly seven behavioural features per student,
matching the fixed feature set used by the OULAD-trained EduGuard-AI
model. Each feature is reconstructed independently within a
(class, student) observation window, so that temporal measures such as
inactivity are never inflated by gaps between unrelated class
enrolments.

| # | Feature | Definition | Educational Interpretation |
|---|---|---|---|
| 1 | `total_clicks` | Sum of `attempt_count` across all interactions | Overall volume of learner interaction activity |
| 2 | `avg_score` | Mean of `correct` across all interactions | Average correctness / academic performance |
| 3 | `active_days` | Count of distinct calendar days with recorded activity | Number of distinct days with active participation |
| 4 | `engagement_variability` | Standard deviation of daily total attempts | Day-to-day inconsistency in learner engagement |
| 5 | `inactivity_days` | (Last active day − first active day) − active days, clipped at zero | Length of inactive periods during the observation window |
| 6 | `engagement_slope` | Ordinary least squares regression of daily attempts against actual elapsed calendar day | Overall temporal trend of learner engagement (increasing vs. declining) |
| 7 | `assessment_consistency` | Standard deviation of `correct` across all interactions | Variation in assessment performance, indicating unstable learning behaviour |

---

## 5. Output Files

### `assistments_student_profiles.csv`

The primary output of this stage of the pipeline. It contains **one row
per student–class observation window**. Each row represents a student's
behavioural profile within a specific class, ensuring that temporal
features are computed independently for each learning context — a
student enrolled in multiple classes will therefore appear as multiple
rows, one per class. Each row includes the following columns:

- `student_id` — unique student identifier (mapped from ASSISTments
  `user_id`)
- `student_class_id` — class-grouping identifier used as the temporal
  observation window boundary
- `total_clicks`
- `avg_score`
- `active_days`
- `engagement_variability`
- `inactivity_days`
- `engagement_slope`
- `assessment_consistency`

This file is fully aligned with the seven-feature schema expected by
the EduGuard-AI behavioural risk model, and is intended as the input
artefact for the downstream external validation and evaluation stages.

---

## 6. Folder Structure

```
ml/
└── external_validation/
    ├── README.md                          # This document
    ├── preprocess_assistments.py          # ASSISTments preprocessing & feature engineering pipeline
    ├── evaluate_external.py               # Not yet implemented (planned)
    ├── compare_results.py                 # Not yet implemented (planned)
    ├── error_analysis.py                  # Not yet implemented (planned)
    ├── computational_benchmark.py         # Not yet implemented (planned)
    │
    ├── figures/                           # Reserved for validation plots and visual diagnostics
    ├── outputs/                           # Reserved for generated model outputs and predictions
    └── reports/
        ├── preprocessing_summary.md       # Completed
        └── external_validation_results.md # Not yet implemented (planned)
```

Dataset artefacts referenced by this pipeline are stored outside this
folder, at:

```
datasets/
└── ASSISTments/
    ├── 2012-2013-data-with-predictions-4-final.csv   # Raw input dataset
    └── assistments_student_profiles.csv              # Generated output (student–class behavioural profiles)
```


---

## 7. Future Pipeline

The following components are planned for subsequent stages of the
external validation pipeline and are **not yet implemented**:

- `evaluate_external.py` — application of the trained EduGuard-AI model
  to `assistments_student_profiles.csv` and computation of external
  performance metrics.
- `compare_results.py` — comparison of external (ASSISTments) validation
  performance against the internal (OULAD) benchmark results.
- `error_analysis.py` — analysis of misclassified students and
  behavioural subgroups on the external dataset.
- `computational_benchmark.py` — measurement of inference-time and
  resource-usage characteristics of the model on external data.

These components will be documented in this README as they are added
to the pipeline.

---

## 8. Version History

### Version 1.0
- Dataset audit completed
- Behavioural feature mapping completed
- ASSISTments preprocessing completed
- Behavioural feature reconstruction completed (`assistments_student_profiles.csv`)

### Version 1.1 (Planned)
- External validation evaluation
- Cross-dataset comparison
- Error analysis
- Computational benchmarking
- SHAP consistency analysis

