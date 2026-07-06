# ASSISTments External Validation — Ground-Truth Supervised Evaluation

## 1. Dataset

- **Source:** ASSISTments 2012-2013 (`2012-2013-data-with-predictions-4-final.csv`), preprocessed by the unmodified `preprocess_assistments.py` into `assistments_student_profiles.csv`.
- **Unit of analysis:** one row per (student_class_id, student_id), matching the granularity the model was designed to score.
- **Records evaluated:** 63174
- **Ground-truth class balance:** 12198 predicted-risk-positive (19.3%) vs. 50976 predicted-risk-negative (80.7%) students (by ground-truth label, not model output).

## 2. Target Variable (Ground Truth)

The raw ASSISTments export contains a real, observed correctness column, `correct` (1 if a student answered a given problem correctly, 0 otherwise). This is genuine ground-truth data, not a label invented for this evaluation.

`correct` is row-level (per problem interaction); the EduGuard-AI risk model predicts at the (student_class_id, student_id) level. `preprocess_assistments.py` already aggregates `correct` into `avg_score = mean(correct) * 100` at exactly this granularity. This evaluation defines:

```
ground_truth_risk = 1  if avg_score < 50
ground_truth_risk = 0  otherwise
```

The 50-point threshold is not a new cutoff invented for this report — it is the same threshold `risk_service.py` already uses elsewhere in this project to flag "Academic performance weakening," reused here rather than introducing an independent decision boundary.

This is a deterministic transformation of real observed data. It is **not** a synthetic label, **not** weak supervision, and **not** fabricated.

## 3. Model

The frozen, OULAD-trained XGBoost psychological-risk classifier (`academic_risk_xgboost.pkl`) is applied as-is, with no retraining or fine-tuning on ASSISTments data, using the same seven fixed behavioural features as training time.

## 4. Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.2034 |
| Precision | 0.1950 |
| Recall | 0.9993 |
| F1-score | 0.3263 |
| ROC-AUC | 0.8556 |

**Confusion Matrix** (rows = ground truth, columns = predicted; see also `figures/assistments_confusion_matrix.png`):

| | Predicted: Low Risk | Predicted: High Risk |
|---|---|---|
| **Actual: Low Risk** | 662 | 50314 |
| **Actual: High Risk** | 9 | 12189 |

**ROC Curve:** see `figures/assistments_roc_curve.png`.

## 5. Interpretation

These metrics indicate how well the OULAD-trained model's risk predictions agree with an ASSISTments-native, correctness-derived risk label, on a population and platform the model was never trained on. Reasonable agreement supports the model's behavioural-risk signal generalizing beyond OULAD; disagreement is equally informative and should be read alongside the distribution-only comparison already reported in `assistments_predictions.csv` / the existing prediction-summary log, not in isolation.

## 6. Limitations

- **Feature/label overlap.** `ground_truth_risk` is derived from `avg_score`, which is *also* one of the seven input features the model consumes. This is a materially different situation from the original OULAD evaluation, where the label (`final_result`) was an outcome column fully independent of the input features. Strong agreement here is partly expected by construction and should not be read as equivalent evidence to an independent held-out test.
- **No true withdrawal/failure outcome.** ASSISTments has no equivalent of OULAD's `final_result` (Withdrawn/Fail/Pass/Distinction). The threshold-based proxy above is the closest available real, observed analogue, not an equivalent construct.
- **Single fixed threshold.** The 50-point cutoff is one reasonable, previously-used-elsewhere choice; results have not been evaluated across alternative thresholds.
- **Class imbalance sensitivity.** Precision/Recall/F1 can be sensitive to the ground-truth class balance reported in Section 1; interpret alongside the confusion matrix, not Accuracy alone.
- **No retraining.** The model is applied frozen; any ASSISTments-specific distribution shift (see the separate prediction-summary log) is not corrected for.

## 7. Threshold Sensitivity Note

The metrics above are computed at a single ground-truth threshold (`avg_score < 50.0`, resulting in the 19.3% / 80.7% class split reported in Section 1). All reported metrics — Accuracy, Precision, Recall, F1, ROC-AUC, and the confusion matrix — are conditional on this specific cutoff, which is configured through a single constant (`GROUND_TRUTH_AVG_SCORE_THRESHOLD`) in `generate_external_predictions.py`. ROC-AUC is threshold-independent with respect to the *decision* threshold on predicted probability, but it is still computed against this one *ground-truth* threshold, so it does not by itself address this sensitivity. A different cutoff would shift the class balance and could shift Precision/Recall/F1 accordingly (Accuracy and ROC-AUC would likely be more stable, but this has not been verified here). Reporting results across a small sweep of alternative thresholds is a natural, low-cost follow-up left for future work, rather than something resolved in this evaluation.

## 8. Reproducibility

Generated by `generate_external_predictions.py`. Inputs: `assistments_student_profiles.csv` (from `preprocess_assistments.py`, unmodified) and `academic_risk_xgboost.pkl` (from `preprocess_oulad.py`, unmodified). No synthetic or weakly-supervised labels were introduced at any step.
