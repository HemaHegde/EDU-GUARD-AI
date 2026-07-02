# SHAP Consistency Report: OULAD vs. ASSISTments

## Purpose

This report evaluates whether the **frozen, OULAD-trained XGBoost psychological
risk model** relies on the same behavioural signals when explained on the
external ASSISTments 2012-2013 population as it does on its own OULAD
training population. It is an **explainability-consistency analysis**,
not a model evaluation: it does not compute, and should not be read as
evidence for or against, predictive accuracy. ASSISTments has no
`psychological_risk` ground truth, so no label-dependent metric
(accuracy, precision, recall, F1, ROC-AUC, calibration, or confusion
matrices) is computed anywhere in this pipeline stage.

## Methodology

### Frozen model usage

A single pre-trained XGBoost classifier (`academic_risk_xgboost.pkl`) was
loaded via `joblib.load()` in inference mode only. The model was **not
retrained, fine-tuned, or refit** at any point in this analysis --
`.fit()` is never called. The exact same in-memory model object is used
to generate SHAP explanations for both populations, so any difference in
attribution patterns reflects a genuine difference in how the model
behaves on the two populations, not a difference in the model itself.

### Same explainer

A single `shap.TreeExplainer` instance was constructed once, wrapping the
frozen model, and reused to compute SHAP values for both the OULAD and
ASSISTments feature matrices. Using one explainer instance for both
populations removes explainer-configuration as a confound in any
observed differences.

### Identical feature space

Both populations are represented with the same fixed seven-feature
behavioural schema that the model was originally trained on, in the same
column order:

- Total Interaction Clicks (`total_clicks`)
- Mean Assessment Score (`avg_score`)
- Active Learning Days (`active_days`)
- Engagement Variability (`engagement_variability`)
- Inactivity Duration (`inactivity_days`)
- Engagement Trend / Slope (`engagement_slope`)
- Assessment Consistency (`assessment_consistency`)

No feature was added, dropped, renamed, or transformed differently
between the two datasets prior to SHAP computation.

### Populations

| Population | Source file | Rows explained |
|---|---|---|
| OULAD (training) | `final_student_psychology_dataset.csv` | 10000 |
| ASSISTments (external) | `assistments_student_profiles.csv` | 10000 |

## Feature Ranking Comparison

Features were ranked 1 (most important) through 7 (least important) in
each population by mean absolute SHAP value.

| Feature | OULAD Rank | ASSISTments Rank | Rank Shift |
|---|---|---|---|
| Inactivity Duration (days) | 1 | 2 | +1 |
| Active Learning Days | 2 | 3 | +1 |
| Mean Assessment Score | 3 | 4 | +1 |
| Assessment Consistency (SD) | 4 | 1 | -3 |
| Engagement Trend (Slope) | 5 | 5 | 0 |
| Total Interaction Clicks | 6 | 6 | 0 |
| Engagement Variability | 7 | 7 | 0 |

## Spearman Rank Correlation

The Spearman rank correlation coefficient between the OULAD and
ASSISTments feature importance rankings is:

**rho = 0.7857** (p = 0.03624)

This corresponds to a **strong** monotonic agreement between
the two rankings. A coefficient near +1 indicates the model draws on
behavioural features in essentially the same relative order of
importance in both populations; a coefficient near 0 or negative would
indicate that the model's reliance on individual features shifts
substantially under population change, independent of whether its
predictions on either population are correct.

## Figures

- `figures/oulad_shap_beeswarm.png` -- per-instance SHAP distribution, OULAD
- `figures/assistments_shap_beeswarm.png` -- per-instance SHAP distribution, ASSISTments
- `figures/oulad_shap_bar.png` -- mean |SHAP| global importance, OULAD
- `figures/assistments_shap_bar.png` -- mean |SHAP| global importance, ASSISTments
- `figures/shap_rank_comparison.png` -- rank slope chart, OULAD vs. ASSISTments

## Scope Note

This report and its underlying script intentionally exclude any
assessment of predictive correctness. Consistency of SHAP attributions
across populations is informative about model behaviour and feature
reliance, but is an entirely separate question from whether the model's
predicted risk labels are accurate for the ASSISTments population, which
this pipeline stage does not and cannot assess without ground-truth
labels.
