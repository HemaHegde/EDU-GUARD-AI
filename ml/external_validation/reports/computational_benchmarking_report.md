# Computational Benchmarking Report

## Purpose

This report documents the computational performance of the **frozen,
OULAD-trained XGBoost psychological risk model** (`academic_risk_xgboost.pkl`)
across model loading, batch prediction, and TreeSHAP explanation, on
both its OULAD training population and the external ASSISTments
population. It is a **resource-usage and latency benchmark**, not a
model evaluation: no accuracy, precision, recall, F1, ROC-AUC,
confusion-matrix, or calibration quantity is computed anywhere in this
script, and none should be inferred from it.

## Methodology

- The frozen model was loaded via `joblib.load()` in inference mode
  only, 10 times in a row, to characterise load-time
  variability. It was never retrained, fine-tuned, or refit; `.fit()`
  is never called anywhere in this script.
- Prediction benchmarking used `model.predict_proba()` on the fixed
  seven-feature matrix for each population (20 timed
  batch runs each, after 1 untimed warm-up run(s) to
  exclude one-off cold-start cost from the reported statistics).
- SHAP benchmarking used a single shared `shap.TreeExplainer` wrapping
  the frozen model (construction timed separately, 5 runs
  each), with value computation timed on a capped, fixed subsample of
  up to 300 rows per population.
- Every timed operation is preceded by an untimed warm-up call; warm-up
  calls are excluded from all reported statistics.
- Memory usage is the process resident-set-size (RSS) delta attributable
  to a single call, in megabytes. CPU usage is the process CPU percent
  accumulated since the previous measurement point, as reported by
  `psutil`.
- All statistics are aggregated as mean, median, standard deviation,
  minimum, and maximum across the repeated runs of each operation.

## Populations Benchmarked

| Population | Rows available | Rows used for prediction benchmark | Rows used for SHAP benchmark |
|---|---|---|---|
| OULAD (training) | 32593 | 20000 | 300 |
| ASSISTments (external) | 63174 | 20000 | 300 |

## System / Environment Information

| Property | Value |
|---|---|
| Platform | Windows-10-10.0.26200-SP0 |
| Python version | 3.11.5 |
| Logical CPU cores | 22 |
| Physical CPU cores | 16 |
| Total system memory | 31.61 GB |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| joblib | 1.5.3 |
| shap | 0.51.0 |
| xgboost | 3.2.0 |
| psutil | 7.2.2 |

## Benchmark Results

| Operation | Dataset | n Runs | Mean (s) | Median (s) | Std (s) | Min (s) | Max (s) | Mean Latency (ms/student) | Mean Throughput (students/sec) | Mean CPU (%) | Mean Memory Δ (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model_loading | not_applicable | 10 | 0.016943 | 0.009186 | 0.024100 | 0.007517 | 0.085429 | n/a | n/a | 114.77 | 2.593 |
| prediction | OULAD | 20 | 0.032814 | 0.031941 | 0.005226 | 0.025743 | 0.042197 | 0.0016 | 623,917.1 | 1321.71 | 0.003 |
| prediction | ASSISTments | 20 | 0.029386 | 0.029321 | 0.001949 | 0.026391 | 0.033174 | 0.0015 | 683,418.7 | 1355.20 | 0.002 |
| shap_explainer_construction | OULAD | 5 | 0.170588 | 0.119471 | 0.078402 | 0.110056 | 0.259107 | n/a | n/a | 778.56 | 1.573 |
| shap_explainer_construction | ASSISTments | 5 | 0.151177 | 0.112908 | 0.059055 | 0.105473 | 0.228520 | n/a | n/a | 911.24 | 0.613 |
| shap_value_computation | OULAD | 5 | 0.122573 | 0.113845 | 0.025663 | 0.107612 | 0.168110 | 0.4086 | 2,517.2 | 1315.54 | 0.001 |
| shap_value_computation | ASSISTments | 5 | 0.109525 | 0.106011 | 0.009557 | 0.103992 | 0.126554 | 0.3651 | 2,754.2 | 1413.40 | 0.002 |

**Reading this table:** "Mean Latency (ms/student)" and "Mean
Throughput (students/sec)" are only meaningful for row-scaled operations
(prediction, SHAP value computation) and are reported as `n/a` for
model loading and SHAP explainer construction, which do not scale with
row count.

## Figures

- `figures/latency_overview.png` -- mean elapsed time (± std) per
  operation, log-scaled
- `figures/throughput_comparison.png` -- mean throughput (students/sec)
  for row-scaled operations
- `figures/memory_cpu_usage.png` -- mean memory delta and mean CPU
  usage per operation

## Scope Note

This report and its underlying script intentionally exclude any
assessment of predictive correctness. All figures describe the
computational cost of running the frozen model's inference and
explanation pipeline -- they say nothing about whether the model's
outputs are accurate for either population, which is outside the scope
of this pipeline stage and cannot be assessed here without ground-truth
labels.
