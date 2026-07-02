# Feature Distribution Comparison Report

`ml/external_validation/reports/feature_distribution_report.md`

## Scope

This report presents a covariate-shift analysis between the OULAD training population and the ASSISTments 2012-2013 external validation population, restricted to the seven fixed behavioural features used by the EduGuard-AI risk model. **This report makes no claims about model accuracy, prediction correctness, or classification performance.** It describes only whether the two populations occupy a similar behavioural feature space.

## Dataset Sizes

- OULAD reference dataset: **32593** rows
- ASSISTments external dataset: **63174** rows

## Descriptive Statistics

Sample size, mean, standard deviation, median, minimum, maximum, and interquartile bounds for each feature, reported separately for each dataset.

| feature                | dataset     |   sample_size |      mean |       std |   median |       min |        max |      p25 |       p75 |
|:-----------------------|:------------|--------------:|----------:|----------:|---------:|----------:|-----------:|---------:|----------:|
| total_clicks           | OULAD       |         32593 | 1479.0334 | 2011.3909 | 758.0000 |    0.0000 | 28615.0000 | 205.0000 | 1990.0000 |
| total_clicks           | ASSISTments |         63174 |  126.1321 |  246.9445 |  46.5000 |    0.0000 |  4938.0000 |  17.0000 |  125.0000 |
| avg_score              | OULAD       |         32593 |   59.7206 |   31.3283 |  71.2857 |    0.0000 |   100.0000 |  50.2857 |   82.2222 |
| avg_score              | ASSISTments |         63174 |   66.1707 |   21.9595 |  69.2308 |    0.0000 |   100.0000 |  53.8275 |   81.5789 |
| active_days            | OULAD       |         32593 |   63.5127 |   58.8844 |  48.0000 |    0.0000 |   286.0000 |  14.0000 |   98.0000 |
| active_days            | ASSISTments |         63174 |    6.4785 |   11.7129 |   3.0000 |    1.0000 |   130.0000 |   1.0000 |    6.0000 |
| engagement_variability | OULAD       |         32593 |    5.0581 |    5.1743 |   4.0058 |    0.0000 |   309.4547 |   2.2679 |    7.0239 |
| engagement_variability | ASSISTments |         63174 |    7.9381 |   12.7647 |   4.5000 |    0.0000 |   440.2350 |   0.0000 |   11.4127 |
| inactivity_days        | OULAD       |         32593 |  113.4976 |   72.2941 | 126.0000 |    0.0000 |   267.0000 |  45.0000 |  176.0000 |
| inactivity_days        | ASSISTments |         63174 |   49.8978 |   74.3999 |   6.0000 |    0.0000 |   344.0000 |   0.0000 |   72.0000 |
| engagement_slope       | OULAD       |         32593 |   -0.0218 |    1.9221 |  -0.0068 | -137.0000 |   123.0000 |  -0.0568 |    0.0269 |
| engagement_slope       | ASSISTments |         63174 |    0.0490 |    9.6939 |   0.0000 | -648.0000 |   500.0000 |  -0.0385 |    0.0342 |
| assessment_consistency | OULAD       |         32593 |   10.3922 |    8.8913 |   9.7882 |    0.0000 |    70.0036 |   0.7071 |   15.9323 |
| assessment_consistency | ASSISTments |         63174 |    0.3953 |    0.1364 |   0.4454 |    0.0000 |     0.5000 |   0.3727 |    0.4845 |

## Distributional Shift Summary (KS Test and PSI)

The two-sample Kolmogorov-Smirnov (KS) test evaluates whether the OULAD and ASSISTments distributions for a feature differ significantly. The Population Stability Index (PSI) quantifies the magnitude of that shift using the following standard interpretation thresholds:

- PSI < 0.10 -- Negligible shift
- 0.10 <= PSI < 0.25 -- Moderate shift
- PSI >= 0.25 -- Significant shift

| feature                |   ks_statistic |   ks_pvalue |    psi | psi_interpretation   |
|:-----------------------|---------------:|------------:|-------:|:---------------------|
| total_clicks           |         0.6058 |      0.0000 | 3.1517 | Significant shift    |
| avg_score              |         0.1639 |      0.0000 | 0.1903 | Moderate shift       |
| active_days            |         0.6508 |      0.0000 | 5.2126 | Significant shift    |
| engagement_variability |         0.2786 |      0.0000 | 1.0898 | Significant shift    |
| inactivity_days        |         0.4317 |      0.0000 | 1.0638 | Significant shift    |
| engagement_slope       |         0.2169 |      0.0000 | 0.6097 | Significant shift    |
| assessment_consistency |         0.7510 |      0.0000 | 8.9017 | Significant shift    |

## Interpretation Notes

- A **negligible or moderate** PSI for a feature suggests that ASSISTments students occupy a broadly comparable region of that feature's distribution to the OULAD training population, which is a precondition for the trained model's behavioural decision logic to be meaningfully applicable to the external population.
- A **significant** PSI for a feature indicates that the two populations differ substantially along that dimension, which should be considered when interpreting any downstream prediction or SHAP-based analysis for that feature.
- These results describe **covariate shift only**. They do not, and cannot, indicate whether the model's predictions on ASSISTments are correct, since no ground-truth psychological risk label exists for that population.
