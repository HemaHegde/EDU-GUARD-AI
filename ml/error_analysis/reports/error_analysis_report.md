# Error Analysis Report -- OULAD Psychological Risk Model

## Scope

This report analyzes the frozen, already-trained OULAD XGBoost psychological risk model on its own held-out OULAD test partition. The model was not retrained, refit, or modified in any way as part of this analysis.

## Dataset and Split

- Full dataset size (all OULAD rows): loaded from `final_student_psychology_dataset.csv`
- Held-out test partition size: **6519** rows
- Split parameters: `test_size=0.20`, `random_state=42`, `stratify=y` (identical to the split used in `preprocess_oulad.py`)

## Evaluation Metrics (Held-Out Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.8891 |
| Precision | 0.9474 |
| Recall | 0.8364 |
| F1 Score | 0.8884 |
| ROC-AUC | 0.9486 |

## Confusion Matrix

| | Predicted Low Risk (0) | Predicted High Risk (1) |
|---|---|---|
| **Actual Low Risk (0)** | 2917 (TN) | 160 (FP) |
| **Actual High Risk (1)** | 563 (FN) | 2879 (TP) |

- Number of false positives: **160**
- Number of false negatives: **563**

## Discussion: Largest Source of Errors

False negatives (563) outnumber false positives (160). The model more often under-predicts psychological risk (missing students who failed or withdrew) than it raises unwarranted alerts.

## Hardest Cases Summary

The 50 hardest cases -- those with predicted probability closest to the 0.5 decision threshold -- represent students whose behavioural feature profile placed them near the model's decision boundary. These are the rows most likely to flip label under small changes in input features or threshold, and are the natural starting point for qualitative review or threshold tuning.

## Output Files

- `outputs/classification_report.csv` -- full per-class precision/recall/F1
- `outputs/confusion_matrix.csv` -- confusion matrix counts
- `outputs/false_positive_cases.csv` -- every false-positive row (all features + probability)
- `outputs/false_negative_cases.csv` -- every false-negative row (all features + probability)
- `outputs/hardest_cases.csv` -- the 50 lowest-confidence predictions
- `figures/confusion_matrix.png`
- `figures/prediction_probability_histogram.png`
- `figures/error_probability_distribution.png`
