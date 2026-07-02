# Prediction Distribution Report: OULAD vs. ASSISTments

## Purpose

This report describes the distribution of predicted probabilities
produced by the **frozen, OULAD-trained XGBoost psychological risk
model** on its own OULAD training population and on the external
ASSISTments 2012-2013 population. It is a **descriptive distributional
analysis**, not a model evaluation, and it does not compute or imply any
label-dependent performance metric. ASSISTments has no
`psychological_risk` ground truth, so accuracy, precision, recall, F1,
ROC-AUC, confusion matrices, and calibration curves are all
out of scope for this pipeline stage and are not computed here.

## Methodology

- The frozen model (`academic_risk_xgboost.pkl`) was loaded via
  `joblib.load()` in inference mode only. It was **not retrained or
  fine-tuned** at any point; `.fit()` is never called.
- OULAD predicted probabilities were generated fresh from the frozen
  model applied to `final_student_psychology_dataset.csv`, using the
  fixed seven-feature schema in the model's expected column order.
- ASSISTments predicted probabilities were loaded directly from the
  pre-computed `assistments_predictions.csv` file produced by
  `generate_external_predictions.py`, and were not regenerated here.
- All statistics below describe the **model's own output distribution**
  and nothing about the correctness of that output.

## Descriptive Statistics of Predicted Probabilities

| Population | n | Mean | Median | Min | Max | Std Dev | Q1 (25th pct) | Q3 (75th pct) | IQR |
|---|---|---|---|---|---|---|---|---|---|
| OULAD | 32593 | 0.5276 | 0.4067 | 0.0092 | 0.9998 | 0.4082 | 0.1128 | 0.9987 | 0.8859 |
| ASSISTments | 63174 | 0.9800 | 0.9981 | 0.0476 | 0.9998 | 0.0835 | 0.9949 | 0.9991 | 0.0042 |

## Risk Group Distribution

Predicted probabilities were bucketed into three descriptive bands:
Low (0.00-0.30), Moderate (0.30-0.70), and High (0.70-1.00). These are
descriptive bins over the model's output, not evaluation thresholds.

### OULAD (training population)

| Risk Group | n Students | % Students |
|---|---|---|
| Low (0.00-0.30) | 14530 | 44.58% |
| Moderate (0.30-0.70) | 4251 | 13.04% |
| High (0.70-1.00) | 13812 | 42.38% |

### ASSISTments (external population)

| Risk Group | n Students | % Students |
|---|---|---|
| Low (0.00-0.30) | 321 | 0.51% |
| Moderate (0.30-0.70) | 750 | 1.19% |
| High (0.70-1.00) | 62103 | 98.30% |

## Fine-Grained Probability Decile Breakdown

For a more granular view of the output distribution than the three-band
risk grouping above, predicted probabilities were also binned into ten
equal-width deciles (0.00-0.10 through 0.90-1.00). This breakdown is
purely descriptive of the model's output and is provided to support
richer reporting tables.

| Probability Decile | OULAD n | OULAD % | ASSISTments n | ASSISTments % |
|---|---|---|---|---|
| 0.00-0.10 | 7290 | 22.37% | 77 | 0.12% |
| 0.10-0.20 | 4738 | 14.54% | 112 | 0.18% |
| 0.20-0.30 | 2502 | 7.68% | 132 | 0.21% |
| 0.30-0.40 | 1666 | 5.11% | 138 | 0.22% |
| 0.40-0.50 | 1153 | 3.54% | 212 | 0.34% |
| 0.50-0.60 | 849 | 2.60% | 194 | 0.31% |
| 0.60-0.70 | 583 | 1.79% | 206 | 0.33% |
| 0.70-0.80 | 502 | 1.54% | 310 | 0.49% |
| 0.80-0.90 | 542 | 1.66% | 1196 | 1.89% |
| 0.90-1.00 | 12768 | 39.17% | 60597 | 95.92% |

## Descriptive Confidence Analysis

A descriptive confidence measure was computed as the distance of each
predicted probability from the 0.5 decision boundary:

```
confidence = abs(predicted_probability - 0.5)
```

Confidence ranges from 0.0 (maximally uncertain output, p = 0.5) to 0.5
(maximally decisive output, p = 0.0 or p = 1.0). **This measure
describes how decisive the model's output is -- it is not, and must
never be interpreted as, a measure of whether that output is correct.**
No ground-truth label is used anywhere in this computation.

### OULAD (training population)

Mean confidence: 0.3863 |
Median confidence: 0.4379

| Confidence Band | n Students | % Students |
|---|---|---|
| Low confidence (0.00-0.15) | 3051 | 9.36% |
| Medium confidence (0.15-0.35) | 6252 | 19.18% |
| High confidence (0.35-0.50) | 23290 | 71.46% |

### ASSISTments (external population)

Mean confidence: 0.4844 |
Median confidence: 0.4981

| Confidence Band | n Students | % Students |
|---|---|---|
| Low confidence (0.00-0.15) | 561 | 0.89% |
| Medium confidence (0.15-0.35) | 989 | 1.57% |
| High confidence (0.35-0.50) | 61624 | 97.55% |

## Figures

- `figures/predicted_probability_distribution.png` -- side-by-side histograms
  of predicted probabilities, OULAD vs. ASSISTments
- `figures/prediction_density_comparison.png` -- overlaid kernel-density
  comparison of predicted probabilities
- `figures/confidence_distribution.png` -- side-by-side histograms of the
  descriptive confidence measure, OULAD vs. ASSISTments

## Data Files

- `outputs/prediction_distribution_statistics.csv` -- mean, median, min,
  max, standard deviation, quartiles, and IQR for both populations
- `outputs/confidence_distribution.csv` -- low/medium/high confidence
  band counts and percentages for both populations
- `outputs/predicted_probability_deciles.csv` -- the ten-bin decile
  breakdown underlying the table above, for both populations

## Scope Note

This report and its underlying script intentionally exclude any
assessment of predictive correctness. Descriptive shifts in the
predicted-probability distribution, risk-group composition, or
confidence distribution between OULAD and ASSISTments are informative
about how the model's output behaves under population change, but are
an entirely separate question from whether those outputs are accurate
for the ASSISTments population, which this pipeline stage does not and
cannot assess without ground-truth labels.
