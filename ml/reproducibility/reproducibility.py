#!/usr/bin/env python3
"""
reproducibility.py
===================

EduGuard-AI — Reproducibility Documentation Module.

This script performs READ-ONLY inspection of the current environment,
codebase, and serialized model artifacts, and produces publication-
quality reproducibility documentation (CSV / JSON / Markdown).

STRICT CONSTRAINTS (enforced by design):
    - Never modifies any backend file.
    - Never retrains a model and never calls .fit().
    - Never performs inference and never calls .predict()/.predict_proba().
    - Never connects to Supabase or any remote database.
    - Never calls the AI Mentor / any LLM / Ollama generation endpoint
      (only `ollama --version` / `ollama list` are invoked, which are
      local metadata commands, not generation calls).
    - Never fabricates a value. Every field is either the result of a
      real inspection, or is explicitly reported as "Not available".

Run directly with:
    python reproducibility.py

No command-line arguments are required or accepted. All paths are
resolved relative to this file's own location, so the script can be
run from any working directory.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("reproducibility")

NOT_AVAILABLE = "Not available"

# Directories that should never be descended into when searching the
# project tree for files (keeps discovery fast and avoids noise).
_EXCLUDED_DIR_NAMES = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build", "site-packages",
}


# ==========================================================================
# SECTION 0 — Project discovery
# ==========================================================================

def discover_project_root(script_path: Path) -> Path:
    """
    Locate the project root relative to this script's own location.

    This script is expected to live at <project_root>/ml/reproducibility/.
    We walk upward from the script location looking for a directory that
    contains a 'backend' folder and/or a 'ml' folder and/or a
    requirements.txt, which are the conventional project-root markers
    for EduGuard-AI. If none is found, we fall back to two levels above
    this file (the conventional ml/reproducibility -> ml -> root path).
    """
    here = script_path.resolve().parent
    candidates = [here] + list(here.parents)

    for candidate in candidates:
        has_backend = (candidate / "backend").is_dir()
        has_ml = (candidate / "ml").is_dir()
        has_requirements = (candidate / "requirements.txt").is_file()
        if has_requirements or (has_backend and has_ml):
            log.info("Project root discovered at: %s", candidate)
            return candidate

    fallback = here.parent.parent if len(here.parents) >= 2 else here
    log.warning(
        "Could not confidently identify project root via markers; "
        "falling back to %s", fallback
    )
    return fallback


def find_file(project_root: Path, filename_patterns: List[str]) -> Optional[Path]:
    """
    Search the project tree for the first file whose name matches any of
    the given patterns (case-insensitive substring match on the file
    name, or exact match). Returns the first match found, preferring
    shallower paths.
    """
    matches: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIR_NAMES]
        for fname in filenames:
            for pattern in filename_patterns:
                if fname.lower() == pattern.lower() or pattern.lower() in fname.lower():
                    matches.append(Path(dirpath) / fname)
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


# ==========================================================================
# SECTION 1 — Environment information
# ==========================================================================

def get_python_info() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }


def get_os_info() -> Dict[str, str]:
    try:
        return {
            "os_system": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "os_platform": platform.platform(),
            "machine_arch": platform.machine(),
        }
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to read OS info: %s", exc)
        return {"os_system": NOT_AVAILABLE}


def get_cpu_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        info["cpu_processor_string"] = platform.processor() or NOT_AVAILABLE
    except Exception:
        info["cpu_processor_string"] = NOT_AVAILABLE

    # Try psutil for accurate core counts.
    try:
        import psutil  # type: ignore
        info["cpu_physical_cores"] = str(psutil.cpu_count(logical=False))
        info["cpu_logical_cores"] = str(psutil.cpu_count(logical=True))
    except Exception:
        info["cpu_physical_cores"] = str(os.cpu_count() or NOT_AVAILABLE)
        info["cpu_logical_cores"] = str(os.cpu_count() or NOT_AVAILABLE)
        log.info("psutil unavailable; using os.cpu_count() for core counts.")

    # On Linux, /proc/cpuinfo often has a more descriptive model name
    # than platform.processor(), which can return an empty string.
    if platform.system() == "Linux":
        try:
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.is_file():
                text = cpuinfo_path.read_text(errors="ignore")
                match = re.search(r"model name\s*:\s*(.+)", text)
                if match:
                    info["cpu_model_name"] = match.group(1).strip()
        except Exception as exc:
            log.info("Could not read /proc/cpuinfo: %s", exc)

    info.setdefault("cpu_model_name", info.get("cpu_processor_string", NOT_AVAILABLE))
    return info


def get_ram_info() -> Dict[str, str]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {
            "ram_total_gb": f"{vm.total / (1024 ** 3):.2f}",
            "ram_available_gb": f"{vm.available / (1024 ** 3):.2f}",
        }
    except Exception:
        log.info("psutil unavailable for RAM info; attempting /proc/meminfo fallback.")
        try:
            if platform.system() == "Linux":
                text = Path("/proc/meminfo").read_text()
                total_kb = int(re.search(r"MemTotal:\s+(\d+) kB", text).group(1))
                avail_match = re.search(r"MemAvailable:\s+(\d+) kB", text)
                result = {"ram_total_gb": f"{total_kb / (1024 ** 2):.2f}"}
                if avail_match:
                    result["ram_available_gb"] = f"{int(avail_match.group(1)) / (1024 ** 2):.2f}"
                return result
        except Exception as exc:
            log.warning("Could not determine RAM info: %s", exc)
        return {"ram_total_gb": NOT_AVAILABLE, "ram_available_gb": NOT_AVAILABLE}


def get_gpu_info() -> Dict[str, str]:
    """
    Attempts to detect GPU hardware via `nvidia-smi`. This is a local
    metadata query, not a network call and not model inference.
    Gracefully reports unavailability if no NVIDIA GPU / driver is found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [ln.strip() for ln in result.stdout.strip().splitlines()]
            return {"gpu_detected": "Yes", "gpu_details": " | ".join(lines)}
        else:
            return {"gpu_detected": "No", "gpu_details": "nvidia-smi returned no GPU data"}
    except FileNotFoundError:
        log.info("nvidia-smi not found; no NVIDIA GPU tooling present.")
        return {"gpu_detected": "No", "gpu_details": "nvidia-smi not installed / no NVIDIA GPU"}
    except Exception as exc:
        log.info("GPU detection failed: %s", exc)
        return {"gpu_detected": "Unknown", "gpu_details": f"Detection error: {exc}"}


# ==========================================================================
# SECTION 2 — Software versions
# ==========================================================================

def get_package_version(package_name: str) -> str:
    """Return the installed version of a package, or NOT_AVAILABLE."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(package_name)
        except PackageNotFoundError:
            return NOT_AVAILABLE
    except Exception as exc:
        log.info("Could not check version for %s: %s", package_name, exc)
        return NOT_AVAILABLE


def get_all_installed_packages() -> List[Dict[str, str]]:
    """Enumerate every installed package and its version, sorted by name."""
    packages: List[Dict[str, str]] = []
    try:
        from importlib.metadata import distributions
        seen = set()
        for dist in distributions():
            try:
                name = dist.metadata["Name"] or dist.metadata.get("Summary", "unknown")
            except Exception:
                name = getattr(dist, "name", None) or "unknown"
            if not name or name in seen:
                continue
            seen.add(name)
            packages.append({"package": name, "version": dist.version or NOT_AVAILABLE})
    except Exception as exc:
        log.warning("Failed to enumerate installed packages: %s", exc)
    packages.sort(key=lambda p: p["package"].lower())
    return packages


def get_requirements_txt_packages(project_root: Path) -> Optional[List[str]]:
    """
    If requirements.txt exists in the project root, return its
    non-comment, non-blank lines. Otherwise return None so the caller
    falls back to inspecting installed packages.
    """
    req_path = project_root / "requirements.txt"
    if not req_path.is_file():
        log.info("requirements.txt not found at %s; will inspect installed packages instead.", req_path)
        return None
    lines = []
    for raw_line in req_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_requirement_name(requirement_line: str) -> Optional[str]:
    """
    Extracts the bare distribution name from a requirements.txt line,
    e.g. 'uvicorn[standard]>=0.29.0' -> 'uvicorn', 'shap>=0.44.0' -> 'shap'.
    Returns None if no plausible package name could be parsed (e.g. a
    line that is a URL, a -r include, or an environment marker only).
    """
    line = requirement_line.strip()
    if not line or line.startswith(("-", "http://", "https://", "git+")):
        return None
    match = re.match(r"^([A-Za-z0-9_.\-]+)", line)
    if not match:
        return None
    return match.group(1)


def get_ollama_info() -> Dict[str, Any]:
    """
    Queries local Ollama metadata only (`ollama --version`, `ollama list`).
    This does NOT call the AI Mentor and does NOT trigger any generation.
    """
    info: Dict[str, Any] = {}
    try:
        version_result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5,
        )
        info["ollama_version"] = version_result.stdout.strip() or version_result.stderr.strip() or NOT_AVAILABLE
    except FileNotFoundError:
        log.info("Ollama executable not found on PATH.")
        info["ollama_version"] = "Ollama not installed"
    except Exception as exc:
        log.info("Failed to query ollama --version: %s", exc)
        info["ollama_version"] = NOT_AVAILABLE

    try:
        list_result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if list_result.returncode == 0:
            info["ollama_models"] = [
                ln.strip() for ln in list_result.stdout.strip().splitlines() if ln.strip()
            ]
        else:
            info["ollama_models"] = ["Ollama list command failed"]
    except FileNotFoundError:
        info["ollama_models"] = ["Ollama not installed"]
    except Exception as exc:
        log.info("Failed to query ollama list: %s", exc)
        info["ollama_models"] = [NOT_AVAILABLE]

    return info


# ==========================================================================
# SECTION 3 — File hashing
# ==========================================================================

def sha256_of_file(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return NOT_AVAILABLE
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:
        log.warning("Failed to hash %s: %s", path, exc)
        return NOT_AVAILABLE


def extract_prompt_template_hash(prompt_builder_path: Optional[Path]) -> Dict[str, str]:
    """
    Computes a SHA-256 hash over the STATIC, argument-free prompt
    section(s) of prompt_builder.py -- i.e. the literal prompt scaffold
    text that does not depend on any runtime StudentContext
    (`_section_system_role` and `_section_response_instructions`, if
    present). This is done via static AST source inspection only: the
    module is NOT imported or executed (its relative imports would
    require the full backend package, and executing arbitrary backend
    code is out of scope for a documentation script).

    If the expected functions cannot be located, this is reported as
    unavailable rather than fabricated or approximated with the whole
    file hash.
    """
    result = {
        "prompt_template_hash": NOT_AVAILABLE,
        "prompt_template_source_functions": NOT_AVAILABLE,
    }
    if prompt_builder_path is None or not prompt_builder_path.is_file():
        log.warning("prompt_builder.py not found; cannot compute prompt template hash.")
        return result

    try:
        source = prompt_builder_path.read_text(errors="ignore")
        tree = ast.parse(source)
        target_names = {"_section_system_role", "_section_response_instructions"}
        found_snippets = []
        found_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in target_names:
                # Only treat as "template" if it takes no data arguments
                # (i.e. it is not parameterized by a runtime context).
                if len(node.args.args) == 0:
                    snippet = ast.get_source_segment(source, node)
                    if snippet:
                        found_snippets.append(snippet)
                        found_names.append(node.name)

        if not found_snippets:
            log.warning(
                "No static, argument-free prompt-template functions found in %s; "
                "prompt template hash left unavailable.", prompt_builder_path
            )
            return result

        combined = "\n\n".join(sorted(found_snippets))
        result["prompt_template_hash"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        result["prompt_template_source_functions"] = ", ".join(sorted(found_names))
        return result
    except Exception as exc:
        log.warning("Failed to statically extract prompt template from %s: %s", prompt_builder_path, exc)
        return result


# ==========================================================================
# SECTION 4 — Model metadata (read-only deserialization; no fit/predict)
# ==========================================================================

def load_pickle_readonly(path: Optional[Path]) -> Any:
    """
    Deserializes a serialized model file for the sole purpose of reading
    stored metadata attributes (hyperparameters, feature names, cluster
    counts, etc). This never calls .fit(), .predict(), .predict_proba(),
    or any inference/training method on the resulting object.

    Model artifacts in this project may be serialized either with plain
    `pickle` or with `joblib` (joblib is the common convention for
    scikit-learn/XGBoost estimators and uses a different framing than
    raw pickle for large numpy arrays). Both are read-only
    deserialization mechanisms, so both are attempted here, in order.
    """
    if path is None or not path.is_file():
        return None

    # Attempt 1: joblib (handles both joblib-native and plain-pickle files)
    try:
        import joblib  # type: ignore
        try:
            return joblib.load(path)
        except ModuleNotFoundError as exc:
            log.warning(
                "Cannot deserialize %s via joblib: required library not "
                "installed (%s). Skipping model-internal metadata for "
                "this artifact.", path.name, exc
            )
            return None
        except Exception as exc:
            log.info("joblib.load failed for %s (%s); trying raw pickle.", path.name, exc)
    except ImportError:
        log.info("joblib not installed; trying raw pickle for %s.", path.name)

    # Attempt 2: raw pickle fallback
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        log.warning(
            "Cannot deserialize %s: required library not installed (%s). "
            "Skipping model-internal metadata for this artifact.", path.name, exc
        )
        return None
    except Exception as exc:
        log.warning("Failed to load %s for metadata inspection: %s", path, exc)
        return None


def inspect_xgboost_model(model_obj: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "xgboost_feature_names": NOT_AVAILABLE,
        "xgboost_hyperparameters": NOT_AVAILABLE,
        "xgboost_random_state": NOT_AVAILABLE,
        "xgboost_n_features_in": NOT_AVAILABLE,
    }
    if model_obj is None:
        return meta

    try:
        if hasattr(model_obj, "get_params"):
            params = model_obj.get_params()
            meta["xgboost_hyperparameters"] = json.dumps(
                {k: v for k, v in params.items() if v is not None}, default=str
            )
            if "random_state" in params:
                meta["xgboost_random_state"] = str(params["random_state"])
    except Exception as exc:
        log.info("Could not read XGBoost get_params(): %s", exc)

    try:
        if hasattr(model_obj, "n_features_in_"):
            meta["xgboost_n_features_in"] = str(model_obj.n_features_in_)
    except Exception:
        pass

    try:
        feature_names = None
        if hasattr(model_obj, "feature_names_in_"):
            feature_names = list(model_obj.feature_names_in_)
        elif hasattr(model_obj, "get_booster"):
            booster = model_obj.get_booster()
            if getattr(booster, "feature_names", None):
                feature_names = list(booster.feature_names)
        if feature_names:
            meta["xgboost_feature_names"] = json.dumps(feature_names)
    except Exception as exc:
        log.info("Could not extract XGBoost feature names: %s", exc)

    return meta


def inspect_sklearn_model(model_obj: Any, label: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        f"{label}_hyperparameters": NOT_AVAILABLE,
        f"{label}_random_state": NOT_AVAILABLE,
        f"{label}_n_features_in": NOT_AVAILABLE,
        f"{label}_feature_names": NOT_AVAILABLE,
    }
    if model_obj is None:
        return meta

    try:
        if hasattr(model_obj, "get_params"):
            params = model_obj.get_params()
            meta[f"{label}_hyperparameters"] = json.dumps(
                {k: v for k, v in params.items() if v is not None}, default=str
            )
            if "random_state" in params:
                meta[f"{label}_random_state"] = str(params["random_state"])
    except Exception as exc:
        log.info("Could not read get_params() for %s: %s", label, exc)

    try:
        if hasattr(model_obj, "n_features_in_"):
            meta[f"{label}_n_features_in"] = str(model_obj.n_features_in_)
    except Exception:
        pass

    try:
        if hasattr(model_obj, "feature_names_in_"):
            meta[f"{label}_feature_names"] = json.dumps(list(model_obj.feature_names_in_))
    except Exception:
        pass

    return meta


# ==========================================================================
# SECTION 5 — Random seed discovery (static source inspection)
# ==========================================================================

def scan_for_random_seeds(*source_paths: Optional[Path]) -> List[str]:
    """
    Statically scans given source files (no execution) for explicit
    'random_state=<int>' or 'seed=<int>' / 'np.random.seed(<int>)'
    assignments as textual evidence of deterministic-seed usage.
    """
    findings: List[str] = []
    pattern = re.compile(
        r"(random_state\s*=\s*\d+|seed\s*=\s*\d+|np\.random\.seed\(\s*\d+\s*\)|"
        r"random\.seed\(\s*\d+\s*\))"
    )
    for path in source_paths:
        if path is None or not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
            matches = sorted(set(pattern.findall(text)))
            for m in matches:
                findings.append(f"{path.name}: {m}")
        except Exception as exc:
            log.info("Could not scan %s for random seeds: %s", path, exc)
    return findings


# ==========================================================================
# SECTION 6 — Output writers
# ==========================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        log.info("Wrote empty CSV (no data available): %s", path)
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Wrote CSV: %s (%d rows)", path, len(rows))


def kv_rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"field": k, "value": v} for k, v in d.items()]


# ==========================================================================
# MAIN
# ==========================================================================

def main() -> None:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    execution_timestamp = datetime.now(timezone.utc).isoformat()

    log.info("=" * 70)
    log.info("EduGuard-AI Reproducibility Documentation Module")
    log.info("Execution timestamp (UTC): %s", execution_timestamp)
    log.info("=" * 70)

    project_root = discover_project_root(script_path)
    cwd = os.getcwd()

    outputs_dir = script_dir / "outputs"
    reports_dir = script_dir / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1-6: Environment
    # ---------------------------------------------------------------
    log.info("Collecting environment information...")
    python_info = get_python_info()
    os_info = get_os_info()
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    gpu_info = get_gpu_info()

    environment: Dict[str, Any] = {}
    environment.update(python_info)
    environment.update(os_info)
    environment.update(cpu_info)
    environment.update(ram_info)
    environment.update(gpu_info)
    environment["working_directory"] = cwd
    environment["execution_timestamp_utc"] = execution_timestamp
    environment["project_root_detected"] = str(project_root)

    # ---------------------------------------------------------------
    # 6-11: Software versions
    # ---------------------------------------------------------------
    log.info("Collecting software / package versions...")
    core_versions = {
        "xgboost_version": get_package_version("xgboost"),
        "tensorflow_version": get_package_version("tensorflow"),
        "scikit_learn_version": get_package_version("scikit-learn"),
        "shap_version": get_package_version("shap"),
    }
    ollama_info = get_ollama_info()

    requirements_lines = get_requirements_txt_packages(project_root)
    if requirements_lines is not None:
        log.info("requirements.txt found with %d entries.", len(requirements_lines))
        installed_packages_rows = []
        for line in requirements_lines:
            pkg_name = parse_requirement_name(line)
            actual_version = get_package_version(pkg_name) if pkg_name else NOT_AVAILABLE
            installed_packages_rows.append({
                "package": pkg_name or line,
                "pinned_requirement": line,
                "installed_version": actual_version,
            })
        package_source = (
            "requirements.txt (package names), cross-referenced against "
            "actually-installed versions via importlib.metadata"
        )
    else:
        all_pkgs = get_all_installed_packages()
        installed_packages_rows = [
            {"package": p["package"], "pinned_requirement": NOT_AVAILABLE,
             "installed_version": p["version"]}
            for p in all_pkgs
        ]
        package_source = "installed environment (importlib.metadata)"
    log.info("Package listing source: %s", package_source)

    # ---------------------------------------------------------------
    # File discovery
    # ---------------------------------------------------------------
    log.info("Locating backend / ML artifact files relative to project root...")
    prompt_builder_path = find_file(project_root, ["prompt_builder.py"])
    mentor_service_path = find_file(project_root, ["mentor_service.py"])
    xgb_model_path = find_file(project_root, ["academic_risk_xgboost.pkl", "xgboost"])
    persona_kmeans_path = find_file(project_root, ["persona_kmeans.pkl", "kmeans"])
    persona_scaler_path = find_file(project_root, ["persona_scaler.pkl", "scaler"])

    for label, p in [
        ("prompt_builder.py", prompt_builder_path),
        ("mentor_service.py", mentor_service_path),
        ("academic_risk_xgboost.pkl", xgb_model_path),
        ("persona_kmeans.pkl", persona_kmeans_path),
        ("persona_scaler.pkl", persona_scaler_path),
    ]:
        log.info("  %-30s -> %s", label, p if p else "NOT FOUND")

    # ---------------------------------------------------------------
    # 13-18: Hashes
    # ---------------------------------------------------------------
    log.info("Computing SHA-256 hashes...")
    prompt_template_info = extract_prompt_template_hash(prompt_builder_path)
    file_hashes = {
        "prompt_template_hash": prompt_template_info["prompt_template_hash"],
        "prompt_template_source_functions": prompt_template_info["prompt_template_source_functions"],
        "prompt_builder_file_hash": sha256_of_file(prompt_builder_path),
        "mentor_service_file_hash": sha256_of_file(mentor_service_path),
        "xgboost_model_hash": sha256_of_file(xgb_model_path),
        "persona_kmeans_model_hash": sha256_of_file(persona_kmeans_path),
        "persona_scaler_model_hash": sha256_of_file(persona_scaler_path),
    }

    # ---------------------------------------------------------------
    # 19-21: Model metadata (read-only deserialization)
    # ---------------------------------------------------------------
    log.info("Loading model artifacts for metadata inspection (no fit/predict)...")
    xgb_model_obj = load_pickle_readonly(xgb_model_path)
    kmeans_model_obj = load_pickle_readonly(persona_kmeans_path)
    scaler_model_obj = load_pickle_readonly(persona_scaler_path)

    xgb_meta = inspect_xgboost_model(xgb_model_obj)
    kmeans_meta = inspect_sklearn_model(kmeans_model_obj, "persona_kmeans")
    scaler_meta = inspect_sklearn_model(scaler_model_obj, "persona_scaler")

    model_metadata: Dict[str, Any] = {}
    model_metadata.update(xgb_meta)
    model_metadata.update(kmeans_meta)
    model_metadata.update(scaler_meta)

    random_seed_findings = scan_for_random_seeds(
        prompt_builder_path, mentor_service_path
    )
    seed_summary: List[str] = list(random_seed_findings)
    for key in ("xgboost_random_state", "persona_kmeans_random_state", "persona_scaler_random_state"):
        val = model_metadata.get(key)
        if val and val != NOT_AVAILABLE:
            seed_summary.append(f"{key} (from serialized model params): {val}")
    if not seed_summary:
        seed_summary = ["No explicit random seeds were detected via static inspection or model params."]

    # ---------------------------------------------------------------
    # Write outputs/environment_summary.csv (single write: environment
    # facts + software versions folded into one at-a-glance file)
    # ---------------------------------------------------------------
    software_versions: Dict[str, Any] = dict(core_versions)
    software_versions["ollama_version"] = ollama_info.get("ollama_version", NOT_AVAILABLE)
    software_versions["ollama_models"] = "; ".join(ollama_info.get("ollama_models", [])) or NOT_AVAILABLE
    write_csv(outputs_dir / "environment_summary.csv",
              kv_rows(environment) + kv_rows(software_versions))

    # ---------------------------------------------------------------
    # Write outputs/installed_packages.csv
    # ---------------------------------------------------------------
    write_csv(outputs_dir / "installed_packages.csv", installed_packages_rows,
              fieldnames=["package", "pinned_requirement", "installed_version"])

    # ---------------------------------------------------------------
    # Write outputs/model_metadata.csv
    # ---------------------------------------------------------------
    write_csv(outputs_dir / "model_metadata.csv", kv_rows(model_metadata))

    # ---------------------------------------------------------------
    # Write outputs/file_hashes.csv
    # ---------------------------------------------------------------
    write_csv(outputs_dir / "file_hashes.csv", kv_rows(file_hashes))

    # ---------------------------------------------------------------
    # Write outputs/reproducibility_summary.json
    # ---------------------------------------------------------------
    summary = {
        "execution_timestamp_utc": execution_timestamp,
        "working_directory": cwd,
        "project_root_detected": str(project_root),
        "environment": environment,
        "software_versions": software_versions,
        "package_listing_source": package_source,
        "installed_packages_count": len(installed_packages_rows),
        "file_hashes": file_hashes,
        "model_metadata": model_metadata,
        "random_seed_findings": seed_summary,
        "discovered_paths": {
            "prompt_builder_py": str(prompt_builder_path) if prompt_builder_path else NOT_AVAILABLE,
            "mentor_service_py": str(mentor_service_path) if mentor_service_path else NOT_AVAILABLE,
            "academic_risk_xgboost_pkl": str(xgb_model_path) if xgb_model_path else NOT_AVAILABLE,
            "persona_kmeans_pkl": str(persona_kmeans_path) if persona_kmeans_path else NOT_AVAILABLE,
            "persona_scaler_pkl": str(persona_scaler_path) if persona_scaler_path else NOT_AVAILABLE,
        },
    }
    with open(outputs_dir / "reproducibility_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote JSON summary: %s", outputs_dir / "reproducibility_summary.json")

    # ---------------------------------------------------------------
    # Write reports/reproducibility_report.md
    # ---------------------------------------------------------------
    write_markdown_report(reports_dir / "reproducibility_report.md", summary)

    log.info("=" * 70)
    log.info("Reproducibility documentation generation complete.")
    log.info("Outputs: %s", outputs_dir)
    log.info("Report:  %s", reports_dir / "reproducibility_report.md")
    log.info("=" * 70)


def write_markdown_report(path: Path, s: Dict[str, Any]) -> None:
    env = s["environment"]
    sw = s["software_versions"]
    fh = s["file_hashes"]
    mm = s["model_metadata"]
    paths = s["discovered_paths"]

    def g(d: Dict[str, Any], key: str) -> str:
        val = d.get(key, NOT_AVAILABLE)
        return str(val) if val not in (None, "") else NOT_AVAILABLE

    lines: List[str] = []
    lines.append("# EduGuard-AI — Reproducibility Report")
    lines.append("")
    lines.append(f"*Automatically generated on {s['execution_timestamp_utc']} (UTC)*")
    lines.append("")

    # 1. Objective
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "This report documents the software environment, hardware "
        "configuration, dependency versions, model artifacts, and prompt "
        "construction logic in effect at the time of generation, for the "
        "purpose of enabling independent reproduction of EduGuard-AI's "
        "risk-prediction and mentoring pipeline results. All values below "
        "were obtained through automatic, read-only inspection of the "
        "live environment and codebase; no value was estimated, assumed, "
        "or hardcoded. Where information could not be determined, this "
        "is stated explicitly rather than inferred."
    )
    lines.append("")

    # 2. Environment
    lines.append("## 2. Environment")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append(f"| Operating System | {g(env, 'os_platform')} |")
    lines.append(f"| OS Release | {g(env, 'os_release')} |")
    lines.append(f"| Machine Architecture | {g(env, 'machine_arch')} |")
    lines.append(f"| CPU Model | {g(env, 'cpu_model_name')} |")
    lines.append(f"| CPU Physical Cores | {g(env, 'cpu_physical_cores')} |")
    lines.append(f"| CPU Logical Cores | {g(env, 'cpu_logical_cores')} |")
    lines.append(f"| Total RAM (GB) | {g(env, 'ram_total_gb')} |")
    lines.append(f"| Available RAM (GB) | {g(env, 'ram_available_gb')} |")
    lines.append(f"| GPU Detected | {g(env, 'gpu_detected')} |")
    lines.append(f"| GPU Details | {g(env, 'gpu_details')} |")
    lines.append(f"| Python Version | {g(env, 'python_version')} ({g(env, 'python_implementation')}) |")
    lines.append(f"| Working Directory (at execution) | {g(env, 'working_directory')} |")
    lines.append(f"| Detected Project Root | {g(env, 'project_root_detected')} |")
    lines.append("")

    # 3. Software Versions
    lines.append("## 3. Software Versions")
    lines.append("")
    lines.append(
        f"Package version information was collected from: "
        f"**{s['package_listing_source']}**. The full installed-package "
        f"listing is available in `outputs/installed_packages.csv`."
    )
    lines.append("")
    lines.append("| Library | Version |")
    lines.append("|---|---|")
    lines.append(f"| XGBoost | {g(sw, 'xgboost_version')} |")
    lines.append(f"| TensorFlow | {g(sw, 'tensorflow_version')} |")
    lines.append(f"| scikit-learn | {g(sw, 'scikit_learn_version')} |")
    lines.append(f"| SHAP | {g(sw, 'shap_version')} |")
    lines.append(f"| Ollama | {g(sw, 'ollama_version')} |")
    lines.append("")
    lines.append(f"**Ollama models detected (`ollama list`):** {g(sw, 'ollama_models')}")
    lines.append("")

    # 4. Model Metadata
    lines.append("## 4. Model Metadata")
    lines.append("")
    lines.append(
        "Metadata below was extracted by deserializing the saved model "
        "artifacts and reading their stored attributes only. No model "
        "was retrained, fitted, or used for inference in the production "
        "of this report."
    )
    lines.append("")
    lines.append("### 4.1 Academic Risk XGBoost Model")
    lines.append("")
    lines.append(f"- **Artifact path:** `{paths.get('academic_risk_xgboost_pkl', NOT_AVAILABLE)}`")
    lines.append(f"- **SHA-256 hash:** `{g(fh, 'xgboost_model_hash')}`")
    lines.append(f"- **Feature names (as stored in model):** {g(mm, 'xgboost_feature_names')}")
    lines.append(f"- **Hyperparameters (get_params()):** {g(mm, 'xgboost_hyperparameters')}")
    lines.append(f"- **random_state:** {g(mm, 'xgboost_random_state')}")
    lines.append("")
    lines.append("### 4.2 Persona KMeans Model")
    lines.append("")
    lines.append(f"- **Artifact path:** `{paths.get('persona_kmeans_pkl', NOT_AVAILABLE)}`")
    lines.append(f"- **SHA-256 hash:** `{g(fh, 'persona_kmeans_model_hash')}`")
    lines.append(f"- **Hyperparameters (get_params()):** {g(mm, 'persona_kmeans_hyperparameters')}")
    lines.append(f"- **random_state:** {g(mm, 'persona_kmeans_random_state')}")
    lines.append("")
    lines.append("### 4.3 Persona Scaler")
    lines.append("")
    lines.append(f"- **Artifact path:** `{paths.get('persona_scaler_pkl', NOT_AVAILABLE)}`")
    lines.append(f"- **SHA-256 hash:** `{g(fh, 'persona_scaler_model_hash')}`")
    lines.append(f"- **Hyperparameters (get_params()):** {g(mm, 'persona_scaler_hyperparameters')}")
    lines.append("")

    # 5. Prompt Reproducibility
    lines.append("## 5. Prompt Reproducibility")
    lines.append("")
    lines.append(f"- **prompt_builder.py path:** `{paths.get('prompt_builder_py', NOT_AVAILABLE)}`")
    lines.append(f"- **prompt_builder.py SHA-256 file hash:** `{g(fh, 'prompt_builder_file_hash')}`")
    lines.append(f"- **mentor_service.py path:** `{paths.get('mentor_service_py', NOT_AVAILABLE)}`")
    lines.append(f"- **mentor_service.py SHA-256 file hash:** `{g(fh, 'mentor_service_file_hash')}`")
    lines.append(
        f"- **Static prompt-template hash:** `{g(fh, 'prompt_template_hash')}` "
        f"(computed over the argument-free scaffold function(s): "
        f"{g(fh, 'prompt_template_source_functions')}). This hash was "
        f"derived via static AST source inspection of "
        f"`prompt_builder.py`; the module was not imported or executed."
    )
    lines.append("")

    # 6. Randomness and Determinism
    lines.append("## 6. Randomness and Determinism")
    lines.append("")
    lines.append(
        "The following random seed evidence was found via (a) static "
        "text scanning of the inspected source files for `random_state=`, "
        "`seed=`, and `random.seed(...)` patterns, and (b) the "
        "`random_state` hyperparameter stored inside the deserialized "
        "model objects, where present."
    )
    lines.append("")
    for finding in s["random_seed_findings"]:
        lines.append(f"- {finding}")
    lines.append("")

    # 7. Reproduction Instructions
    lines.append("## 7. Reproduction Instructions")
    lines.append("")
    lines.append(
        "1. Recreate the software environment using the versions listed "
        "in Section 3 and `outputs/installed_packages.csv` "
        f"(source: {s['package_listing_source']})."
    )
    lines.append(
        "2. Verify artifact integrity by recomputing the SHA-256 hashes "
        "in `outputs/file_hashes.csv` for `prompt_builder.py`, "
        "`mentor_service.py`, and the three model `.pkl` files, and "
        "confirming they match the values in this report."
    )
    lines.append(
        "3. If any hash differs, the corresponding artifact has changed "
        "since this report was generated and results may not be "
        "reproducible until the artifact is reverted or a new "
        "reproducibility report is generated."
    )
    lines.append(
        "4. If GPU acceleration was reported as unavailable in Section 2, "
        "results were produced on CPU only; using a GPU-equipped machine "
        "may change floating-point summation order and, in rare cases, "
        "numerical outputs at the last significant digits."
    )
    lines.append(
        "5. Re-run this script (`python reproducibility.py`, no arguments) "
        "from within the project tree at any later point in time to "
        "generate a fresh, comparable reproducibility snapshot."
    )
    lines.append("")

    # 8. Limitations
    lines.append("## 8. Limitations")
    lines.append("")
    lines.append(
        "- This report reflects the state of the environment and "
        "artifacts **at the moment it was generated**; it is not a "
        "historical record of prior training runs unless those runs "
        "produced the exact artifacts inspected here."
    )
    lines.append(
        "- Model hyperparameters and feature names are only as complete "
        "as what scikit-learn/XGBoost chose to serialize into the "
        "pickled object; parameters set only at `fit()`-time internals "
        "that are not exposed as attributes cannot be recovered without "
        "retraining, which this script explicitly does not do."
    )
    lines.append(
        "- The prompt-template hash covers only the static, "
        "context-independent scaffold sections of `prompt_builder.py`; "
        "the fully assembled runtime prompt also depends on live student "
        "context, SHAP output, cognitive state, and retrieved material, "
        "none of which this script computes or accesses."
    )
    lines.append(
        "- Any field listed as \"Not available\" indicates the "
        "corresponding tool, file, or library was not present/importable "
        "in the environment this report was generated in, or is not a "
        "risk artifact. It is intentionally not a fabricated value."
    )
    lines.append("")

    # 9. Reproducibility Checklist — each box reflects whether the
    # corresponding value was ACTUALLY captured above, computed from the
    # real collected data (not asserted).
    lines.append("## 9. Reproducibility Checklist")
    lines.append("")
    lines.append(
        "Each item below is checked only if the corresponding artifact "
        "was actually inspected and a real value was captured above; "
        "unchecked items indicate the information was unavailable in "
        "this run (see Section 8 for why)."
    )
    lines.append("")

    def present(val: str) -> bool:
        return val not in (None, "", NOT_AVAILABLE) and "not available" not in str(val).lower()

    has_seed_evidence = bool(s["random_seed_findings"]) and not (
        len(s["random_seed_findings"]) == 1
        and s["random_seed_findings"][0].startswith("No explicit random seeds")
    )
    has_feature_names = present(g(mm, "xgboost_feature_names")) or present(g(mm, "persona_scaler_feature_names"))
    all_file_hashes_present = all(
        present(fh.get(k, NOT_AVAILABLE))
        for k in ("prompt_builder_file_hash", "mentor_service_file_hash",
                   "xgboost_model_hash", "persona_kmeans_model_hash",
                   "persona_scaler_model_hash")
    )

    checklist = [
        ("Frozen XGBoost model artifact hashed", present(g(fh, "xgboost_model_hash"))),
        ("Frozen Persona (KMeans) model artifact hashed", present(g(fh, "persona_kmeans_model_hash"))),
        ("Frozen Scaler artifact hashed", present(g(fh, "persona_scaler_model_hash"))),
        ("Prompt template hash captured", present(g(fh, "prompt_template_hash"))),
        ("Mentor service file hash captured", present(g(fh, "mentor_service_file_hash"))),
        ("Python version recorded", present(g(env, "python_version"))),
        ("Package versions recorded", int(s.get("installed_packages_count", 0)) > 0),
        ("Hardware (CPU/RAM/GPU) recorded", present(g(env, "cpu_model_name")) and present(g(env, "ram_total_gb"))),
        ("Random seed evidence found", has_seed_evidence),
        ("Model feature names captured", has_feature_names),
        ("All SHA-256 file hashes captured", all_file_hashes_present),
    ]
    for label, ok in checklist:
        box = "[x]" if ok else "[ ]"
        lines.append(f"- {box} {label}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote Markdown report: %s", path)


if __name__ == "__main__":
    main()
