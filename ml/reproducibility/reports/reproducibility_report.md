# EduGuard-AI — Reproducibility Report

*Automatically generated on 2026-07-02T05:52:49.481884+00:00 (UTC)*

## 1. Objective

This report documents the software environment, hardware configuration, dependency versions, model artifacts, and prompt construction logic in effect at the time of generation, for the purpose of enabling independent reproduction of EduGuard-AI's risk-prediction and mentoring pipeline results. All values below were obtained through automatic, read-only inspection of the live environment and codebase; no value was estimated, assumed, or hardcoded. Where information could not be determined, this is stated explicitly rather than inferred.

## 2. Environment

| Property | Value |
|---|---|
| Operating System | Windows-10-10.0.26200-SP0 |
| OS Release | 10 |
| Machine Architecture | AMD64 |
| CPU Model | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel |
| CPU Physical Cores | 16 |
| CPU Logical Cores | 22 |
| Total RAM (GB) | 31.61 |
| Available RAM (GB) | 15.64 |
| GPU Detected | No |
| GPU Details | nvidia-smi not installed / no NVIDIA GPU |
| Python Version | 3.11.5 (CPython) |
| Working Directory (at execution) | C:\Users\Anika\Downloads\EDU-GUARD-AI-main\ml\reproducibility |
| Detected Project Root | C:\Users\Anika\Downloads\EDU-GUARD-AI-main |

## 3. Software Versions

Package version information was collected from: **requirements.txt (package names), cross-referenced against actually-installed versions via importlib.metadata**. The full installed-package listing is available in `outputs/installed_packages.csv`.

| Library | Version |
|---|---|
| XGBoost | 3.2.0 |
| TensorFlow | 2.21.0 |
| scikit-learn | 1.9.0 |
| SHAP | 0.51.0 |
| Ollama | ollama version is 0.7.1 |

**Ollama models detected (`ollama list`):** NAME                   ID              SIZE      MODIFIED; qwen2.5:3b             357c53fb659c    1.9 GB    3 weeks ago; mistral:latest         6577803aa9a0    4.4 GB    3 weeks ago; health:latest          a0e5922867a8    4.7 GB    12 months ago; llama3:latest          365c0bd3c000    4.7 GB    12 months ago; diet-mistral:latest    616cc265ca5d    4.1 GB    13 months ago

## 4. Model Metadata

Metadata below was extracted by deserializing the saved model artifacts and reading their stored attributes only. No model was retrained, fitted, or used for inference in the production of this report.

### 4.1 Academic Risk XGBoost Model

- **Artifact path:** `C:\Users\Anika\Downloads\EDU-GUARD-AI-main\ml\academic_risk_xgboost.pkl`
- **SHA-256 hash:** `216342fbd7c829e683eff84789c8c23cc294c7b667aaea5b7a6a82a19f898bb4`
- **Feature names (as stored in model):** ["total_clicks", "avg_score", "active_days", "engagement_variability", "inactivity_days", "engagement_slope", "assessment_consistency"]
- **Hyperparameters (get_params()):** {"objective": "binary:logistic", "colsample_bytree": 0.8, "enable_categorical": false, "eval_metric": "logloss", "learning_rate": 0.03, "max_depth": 6, "missing": NaN, "n_estimators": 300, "random_state": 42, "subsample": 0.8, "verbosity": 0}
- **random_state:** 42

### 4.2 Persona KMeans Model

- **Artifact path:** `C:\Users\Anika\Downloads\EDU-GUARD-AI-main\ml\persona_kmeans.pkl`
- **SHA-256 hash:** `3d8b3bcd827b74ee2f0e49d4a225ef5be8c2772774b4cd94cc69058aa2a2ad44`
- **Hyperparameters (get_params()):** {"algorithm": "lloyd", "copy_x": true, "init": "k-means++", "max_iter": 300, "n_clusters": 6, "n_init": 10, "random_state": 42, "tol": 0.0001, "verbose": 0}
- **random_state:** 42

### 4.3 Persona Scaler

- **Artifact path:** `C:\Users\Anika\Downloads\EDU-GUARD-AI-main\ml\persona_scaler.pkl`
- **SHA-256 hash:** `15e968591136d4d045573bea5e8d1acb85c204da6b29b9a48a39cd57bd95f4b4`
- **Hyperparameters (get_params()):** {"copy": true, "with_mean": true, "with_std": true}

## 5. Prompt Reproducibility

- **prompt_builder.py path:** `C:\Users\Anika\Downloads\EDU-GUARD-AI-main\backend\services\prompt_builder.py`
- **prompt_builder.py SHA-256 file hash:** `2893eeae542ee7921589efe4f71cdf859db728459af3ab8b7b306d4d1c7d2095`
- **mentor_service.py path:** `C:\Users\Anika\Downloads\EDU-GUARD-AI-main\backend\services\mentor_service.py`
- **mentor_service.py SHA-256 file hash:** `54c3e2d7a37b287075c82a436a9bbd131322dc0108f7ecebde62af1133032e6b`
- **Static prompt-template hash:** `3661846a5142b03245b4a63b5aeb38648c6cee397ad79e2119cef6f14b8029d2` (computed over the argument-free scaffold function(s): _section_response_instructions, _section_system_role). This hash was derived via static AST source inspection of `prompt_builder.py`; the module was not imported or executed.

## 6. Randomness and Determinism

The following random seed evidence was found via (a) static text scanning of the inspected source files for `random_state=`, `seed=`, and `random.seed(...)` patterns, and (b) the `random_state` hyperparameter stored inside the deserialized model objects, where present.

- xgboost_random_state (from serialized model params): 42
- persona_kmeans_random_state (from serialized model params): 42

## 7. Reproduction Instructions

1. Recreate the software environment using the versions listed in Section 3 and `outputs/installed_packages.csv` (source: requirements.txt (package names), cross-referenced against actually-installed versions via importlib.metadata).
2. Verify artifact integrity by recomputing the SHA-256 hashes in `outputs/file_hashes.csv` for `prompt_builder.py`, `mentor_service.py`, and the three model `.pkl` files, and confirming they match the values in this report.
3. If any hash differs, the corresponding artifact has changed since this report was generated and results may not be reproducible until the artifact is reverted or a new reproducibility report is generated.
4. If GPU acceleration was reported as unavailable in Section 2, results were produced on CPU only; using a GPU-equipped machine may change floating-point summation order and, in rare cases, numerical outputs at the last significant digits.
5. Re-run this script (`python reproducibility.py`, no arguments) from within the project tree at any later point in time to generate a fresh, comparable reproducibility snapshot.

## 8. Limitations

- This report reflects the state of the environment and artifacts **at the moment it was generated**; it is not a historical record of prior training runs unless those runs produced the exact artifacts inspected here.
- Model hyperparameters and feature names are only as complete as what scikit-learn/XGBoost chose to serialize into the pickled object; parameters set only at `fit()`-time internals that are not exposed as attributes cannot be recovered without retraining, which this script explicitly does not do.
- The prompt-template hash covers only the static, context-independent scaffold sections of `prompt_builder.py`; the fully assembled runtime prompt also depends on live student context, SHAP output, cognitive state, and retrieved material, none of which this script computes or accesses.
- Any field listed as "Not available" indicates the corresponding tool, file, or library was not present/importable in the environment this report was generated in, or is not a risk artifact. It is intentionally not a fabricated value.

## 9. Reproducibility Checklist

Each item below is checked only if the corresponding artifact was actually inspected and a real value was captured above; unchecked items indicate the information was unavailable in this run (see Section 8 for why).

- [x] Frozen XGBoost model artifact hashed
- [x] Frozen Persona (KMeans) model artifact hashed
- [x] Frozen Scaler artifact hashed
- [x] Prompt template hash captured
- [x] Mentor service file hash captured
- [x] Python version recorded
- [x] Package versions recorded
- [x] Hardware (CPU/RAM/GPU) recorded
- [x] Random seed evidence found
- [x] Model feature names captured
- [x] All SHA-256 file hashes captured
