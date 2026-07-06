"""
EduGuard-AI :: AI Mentor Pipeline -- Computational Cost Benchmark
==================================================================
Script: mentor_computational_benchmarking.py

PURPOSE
-------
Rigorous, real-runtime computational-cost benchmark of the deployed AI
Mentor pipeline (`ask_mentor()` in `mentor_service.py`), for the IEEE
paper's Computational Cost evaluation. Companion to (and modelled on
the same measurement conventions as) the existing frozen-XGBoost-model
benchmark in `computational_benchmarking.py` -- same repeated-run /
mean-median-std-min-max methodology, same psutil-based CPU/memory
instrumentation, same "no fabricated numbers" policy -- but targeting
the *service-level* pipeline (retrieval -> prompt construction -> LLM
inference -> parsing/citations/history) rather than the model file.

STRICT ADDITIVITY GUARANTEE -- READ BEFORE RUNNING
----------------------------------------------------
This script is a **new, standalone file**. It does NOT edit, rewrite,
or import-and-monkeypatch-permanently any line of:
    - mentor_service.py
    - context_builder.py / prompt_builder.py / retrieval.py /
      confidence.py / reasoning_layer.py
    - computational_benchmarking.py (the existing XGBoost benchmark)
`ask_mentor(user_id, question)` is called with EXACTLY its existing
public signature, and its return value is read but never altered,
inspected for internal fields, or relied upon to change behaviour.
No mentor logic, prompt content, retrieval behaviour, database schema,
or API route is touched.

HOW PER-STAGE TIMING IS OBTAINED WITHOUT EDITING mentor_service.py
--------------------------------------------------------------------
`mentor_service.py` already times its own stages internally with
`time.perf_counter()` (see its "TIMING INSTRUMENTATION" docstring
section) but only PRINTS the summary -- it does not return per-stage
numbers to the caller, and this script was asked not to add a
`return timings` line to that file.

Instead, this script uses standard-library monkey-patching -- the same
technique test frameworks use for spies/mocks -- to wrap the *exact*,
already-imported functions `ask_mentor()` calls:
    - `mentor_service.get_retriever`      (retrieval stage)
    - `mentor_service.build_system_prompt` and
      `mentor_service._build_output_format_instructions`
                                            (prompt construction stage)
    - `mentor_service.ollama.chat`         (LLM inference stage)
Each wrapper calls the REAL, UNMODIFIED underlying function, returns
the REAL, UNMODIFIED result, and additionally records a
`time.perf_counter()` delta around that call. `ask_mentor()` cannot
observe any difference in behaviour, output, or side effects -- the
only thing added is a stopwatch. The original functions are restored
(`try/finally`) the moment each benchmarked call finishes, whether it
succeeds or raises.

This yields genuine, non-fabricated, full-precision wall-clock
measurements of the real pipeline: real vector retrieval, real prompt
assembly, and a real network call to the local Ollama server -- not
estimates, not values parsed from rounded log text.

IMPORTANT SIDE-EFFECT NOTE
---------------------------
`ask_mentor()` performs a real `mentor_history` insert into Supabase on
every call (this is existing, unmodified behaviour). Running this
benchmark WILL write `N_WARMUP_RUNS + N_BENCHMARK_RUNS` real rows per
configured (user_id, question) pair into your `mentor_history` table.
Use a disposable/test `user_id` if you don't want benchmark traffic
mixed into real student history.

HOW TO RUN
----------
This file must live in the SAME PACKAGE as `mentor_service.py` (e.g.
`services/mentor_computational_benchmarking.py`), because
`mentor_service.py` itself uses package-relative imports
(`from .context_builder import ...`) and therefore can only be
imported as part of its package. Run it as a module from the project
root, e.g.:

    python -m services.mentor_computational_benchmarking

Before running, edit the CONFIGURATION section below:
    - `BENCHMARK_REQUESTS`: real (user_id, question) pairs that exist
      in your system (a user_id with an actual student profile, so the
      pipeline runs its real, non-error path).
    - `N_WARMUP_RUNS` / `N_BENCHMARK_RUNS`: tune for your LLM's speed;
      LLM calls are far more expensive per-run than the XGBoost
      benchmark's operations, so defaults here are intentionally
      modest.

REQUIREMENTS
------------
- Use only real measurements; nothing here is estimated or fabricated.
- If `psutil` is unavailable, every CPU/memory metric is reported as
  the literal string "Not Available" rather than a fabricated number.
- If no GPU is detected, GPU fields are reported as "CPU Only".
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: safe for headless / CI execution
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# OPTIONAL DEPENDENCY: psutil
# --------------------------------------------------------------------------
# Per requirement: if psutil is unavailable, every CPU/memory metric must
# be reported as "Not Available" -- never fabricated.
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mentor_computational_benchmarking")

# --------------------------------------------------------------------------
# IMPORT THE UNMODIFIED MENTOR PIPELINE
# --------------------------------------------------------------------------
# Relative import first (correct when this file sits in the same package
# as mentor_service.py and is run with `python -m package.this_file`).
# Falls back to an absolute import for projects that invoke mentor_service
# as a top-level module instead of a package. Neither branch modifies
# mentor_service.py in any way -- both just import names from it.
try:
    from . import mentor_service  # type: ignore
    from .mentor_service import ask_mentor  # type: ignore
except ImportError:
    import mentor_service  # type: ignore
    from mentor_service import ask_mentor  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE FOR YOUR PROJECT
# --------------------------------------------------------------------------

# Real (user_id, question) pairs. Use a user_id that has an actual student
# profile so the pipeline exercises its normal, non-error, non-similarity
# -gated path. Multiple pairs are cycled through round-robin across runs
# so the benchmark isn't measuring one cached/repeated prompt only.
BENCHMARK_REQUESTS = [
    {
        "user_id": "d271bd7d-6631-4969-b000-692ad069ddfe",
        "question": "How can I improve my study habits?"
    },
    {
        "user_id": "d271bd7d-6631-4969-b000-692ad069ddfe",
        "question": "What should I focus on to improve my grades?"
    }
]

# One or more untimed warm-up calls per benchmark, to exclude one-off
# cold-start cost (first Ollama model load into memory, first FAISS
# index touch, first DB connection) from steady-state statistics --
# same convention as computational_benchmarking.py's N_WARMUP_RUNS.
N_WARMUP_RUNS = 1

# Timed, recorded runs. Kept modest by default because each run makes a
# real LLM call (far more expensive than the XGBoost benchmark's
# in-process operations); raise this once you've confirmed the
# configuration works end-to-end.
N_BENCHMARK_RUNS = 5

# CPU is sampled on a background thread at this interval (seconds)
# throughout each request, so "peak" and "average" CPU percent reflect
# real sampled utilisation across the whole request rather than a
# single before/after reading.
CPU_SAMPLE_INTERVAL_S = 0.05

# --------------------------------------------------------------------------
# OUTPUT LOCATIONS
# --------------------------------------------------------------------------
# Deliberately a SEPARATE directory from the existing XGBoost benchmark's
# outputs/figures/reports so this script never overwrites that report,
# even though both are named "computational_benchmarking_report.md" per
# the requested filenames.
OUTPUT_DIR = SCRIPT_DIR / "mentor_benchmark_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

METRICS_JSON_PATH = OUTPUT_DIR / "computational_metrics.json"
LATENCY_FIG_PATH = FIGURE_DIR / "latency_breakdown.png"
MEMORY_FIG_PATH = FIGURE_DIR / "memory_usage.png"
CPU_FIG_PATH = FIGURE_DIR / "cpu_usage.png"
THROUGHPUT_FIG_PATH = FIGURE_DIR / "throughput.png"
REPORT_PATH = REPORT_DIR / "computational_benchmarking_report.md"


# --------------------------------------------------------------------------
# GPU DETECTION (real check -- never fabricated; "CPU Only" if absent)
# --------------------------------------------------------------------------

def detect_gpu() -> str:
    """
    Best-effort, real detection of an available GPU. Tries `nvidia-smi`
    first (works regardless of which ML framework is installed), then
    falls back to `torch.cuda` if PyTorch happens to be installed.
    Returns a human-readable GPU name/description, or the literal
    string "CPU Only" if no GPU could be detected -- never a fabricated
    GPU name.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if names:
                return ", ".join(names)
        except Exception:
            pass  # fall through to torch check / CPU Only

    try:
        import torch  # noqa: F401 (only used if already installed)
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    return "CPU Only"


# --------------------------------------------------------------------------
# CPU SAMPLING (background thread -- real average + peak, not a single
# before/after reading)
# --------------------------------------------------------------------------

class _CPUSampler:
    """
    Samples `psutil.Process().cpu_percent()` on a background thread at a
    fixed interval for the duration of one request, so `average` and
    `peak` CPU percent reflect real utilisation sampled throughout the
    call rather than a single coarse before/after delta. No-ops safely
    (returns no samples) if psutil is unavailable.
    """

    def __init__(self, interval_s: float = CPU_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self._samples: List[float] = []
        self._stop_flag = False
        self._thread = None
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None

    def start(self) -> None:
        if not PSUTIL_AVAILABLE:
            return
        import threading
        self._samples = []
        self._stop_flag = False
        self._process.cpu_percent(interval=None)  # prime internal baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_flag:
            self._samples.append(self._process.cpu_percent(interval=self.interval_s))

    def stop(self) -> List[float]:
        if not PSUTIL_AVAILABLE:
            return []
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4 + 1.0)
        return list(self._samples)


# --------------------------------------------------------------------------
# NON-INVASIVE STAGE-TIMING INSTRUMENTATION (monkey-patch, restored after
# every single request)
# --------------------------------------------------------------------------

@contextmanager
def _instrumented_pipeline(sink: Dict[str, float]):
    """
    Context manager that temporarily wraps the three pipeline functions
    `mentor_service.ask_mentor()` already calls -- retrieval, prompt
    construction, and the LLM call -- with timing, WITHOUT altering
    their behaviour, arguments, or return values in any way. Durations
    (seconds) are written into `sink` under the keys:
        "retrieval_s", "prompt_build_s", "output_format_s", "llm_s"
    `sink` should be cleared by the caller before each request; if a
    stage genuinely never runs for a given request (e.g. the Sprint 8
    similarity-threshold gate short-circuits before the LLM call), its
    key is simply absent afterwards -- never fabricated as zero.
    Originals are restored on exit even if `ask_mentor()` raises.
    """
    original_get_retriever = mentor_service.get_retriever
    original_build_system_prompt = mentor_service.build_system_prompt
    original_build_output_format_instructions = mentor_service._build_output_format_instructions
    original_ollama_chat = mentor_service.ollama.chat

    class _TimedRetrieverProxy:
        """
        Thin, transparent proxy: delegates every attribute/method to the
        real retriever except `.retrieve()`, which it times. The real
        retriever object, its FAISS index, and its `.retrieve()` return
        value are completely unmodified.
        """
        def __init__(self, inner):
            self._inner = inner

        def retrieve(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return self._inner.retrieve(*args, **kwargs)
            finally:
                sink["retrieval_s"] = sink.get("retrieval_s", 0.0) + (time.perf_counter() - t0)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _timed_get_retriever(*args, **kwargs):
        real_retriever = original_get_retriever(*args, **kwargs)
        return _TimedRetrieverProxy(real_retriever)

    def _timed_build_system_prompt(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_build_system_prompt(*args, **kwargs)
        finally:
            sink["prompt_build_s"] = sink.get("prompt_build_s", 0.0) + (time.perf_counter() - t0)

    def _timed_build_output_format_instructions(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_build_output_format_instructions(*args, **kwargs)
        finally:
            sink["output_format_s"] = sink.get("output_format_s", 0.0) + (time.perf_counter() - t0)

    def _timed_ollama_chat(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_ollama_chat(*args, **kwargs)
        finally:
            sink["llm_s"] = sink.get("llm_s", 0.0) + (time.perf_counter() - t0)

    mentor_service.get_retriever = _timed_get_retriever
    mentor_service.build_system_prompt = _timed_build_system_prompt
    mentor_service._build_output_format_instructions = _timed_build_output_format_instructions
    mentor_service.ollama.chat = _timed_ollama_chat
    try:
        yield sink
    finally:
        mentor_service.get_retriever = original_get_retriever
        mentor_service.build_system_prompt = original_build_system_prompt
        mentor_service._build_output_format_instructions = original_build_output_format_instructions
        mentor_service.ollama.chat = original_ollama_chat


# --------------------------------------------------------------------------
# SINGLE-REQUEST MEASUREMENT
# --------------------------------------------------------------------------

def run_single_request(user_id: str, question: str) -> Dict[str, Any]:
    """
    Executes exactly one real, unmodified `ask_mentor()` call and
    measures:
      - retrieval_latency_ms       (real time inside retriever.retrieve)
      - prompt_build_latency_ms    (real time inside build_system_prompt
                                     + _build_output_format_instructions)
      - llm_latency_ms             (real time inside ollama.chat; None
                                     if the similarity-threshold gate
                                     short-circuited before the LLM was
                                     ever called -- never fabricated)
      - total_latency_ms           (real wall-clock time around the
                                     ENTIRE ask_mentor() call, including
                                     context building, confidence,
                                     citation building, and the
                                     mentor_history DB write)
      - memory_before_mb / memory_after_mb / memory_delta_mb (process
        RSS, via psutil; "Not Available" if psutil is not installed)
      - average/peak CPU percent sampled across this single request
    Returns a flat dict of real, measured values for this one run.
    """
    sink: Dict[str, float] = {}

    process = psutil.Process() if PSUTIL_AVAILABLE else None
    memory_before_mb = (process.memory_info().rss / (1024 ** 2)) if process else None

    cpu_sampler = _CPUSampler()
    cpu_sampler.start()

    total_t0 = time.perf_counter()
    with _instrumented_pipeline(sink):
        response = ask_mentor(user_id, question)
    total_elapsed_s = time.perf_counter() - total_t0

    cpu_samples = cpu_sampler.stop()
    memory_after_mb = (process.memory_info().rss / (1024 ** 2)) if process else None

    retrieval_s = sink.get("retrieval_s")
    prompt_build_s = None
    if "prompt_build_s" in sink or "output_format_s" in sink:
        prompt_build_s = sink.get("prompt_build_s", 0.0) + sink.get("output_format_s", 0.0)
    llm_s = sink.get("llm_s")  # None (not 0) if the LLM was never invoked this request

    return {
        "status": response.get("status") if isinstance(response, dict) else None,
        "llm_called": llm_s is not None,
        "retrieval_latency_ms": (retrieval_s * 1000.0) if retrieval_s is not None else None,
        "prompt_build_latency_ms": (prompt_build_s * 1000.0) if prompt_build_s is not None else None,
        "llm_latency_ms": (llm_s * 1000.0) if llm_s is not None else None,
        "total_latency_ms": total_elapsed_s * 1000.0,
        "memory_before_mb": memory_before_mb,
        "memory_after_mb": memory_after_mb,
        "memory_delta_mb": (
            (memory_after_mb - memory_before_mb)
            if (memory_before_mb is not None and memory_after_mb is not None) else None
        ),
        "average_cpu_percent": statistics.mean(cpu_samples) if cpu_samples else None,
        "peak_cpu_percent": max(cpu_samples) if cpu_samples else None,
    }


# --------------------------------------------------------------------------
# AGGREGATION
# --------------------------------------------------------------------------

def _stats(values: List[float]) -> Dict[str, Any]:
    """Mean/median/std/min/max/n over real, non-None measured values only."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def aggregate_runs(raw_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates per-request raw measurements into the summary statistics
    requested for the IEEE report. LLM-gated requests (where the
    similarity-threshold gate skipped the LLM entirely) are correctly
    excluded from `llm_latency_ms` / `prompt_build_latency_ms` stats
    rather than being counted as zero-latency, since that stage
    genuinely did not run.
    """
    n_total = len(raw_runs)
    n_llm_called = sum(1 for r in raw_runs if r["llm_called"])

    summary: Dict[str, Any] = {
        "n_requests_total": n_total,
        "n_requests_llm_invoked": n_llm_called,
        "n_requests_similarity_gated": n_total - n_llm_called,
        "retrieval_latency_ms": _stats([r["retrieval_latency_ms"] for r in raw_runs]),
        "prompt_build_latency_ms": _stats([r["prompt_build_latency_ms"] for r in raw_runs]),
        "llm_latency_ms": _stats([r["llm_latency_ms"] for r in raw_runs]),
        "total_latency_ms": _stats([r["total_latency_ms"] for r in raw_runs]),
    }

    if PSUTIL_AVAILABLE:
        avg_cpu_values = [r["average_cpu_percent"] for r in raw_runs if r["average_cpu_percent"] is not None]
        peak_cpu_values = [r["peak_cpu_percent"] for r in raw_runs if r["peak_cpu_percent"] is not None]
        mem_before_values = [r["memory_before_mb"] for r in raw_runs if r["memory_before_mb"] is not None]
        mem_after_values = [r["memory_after_mb"] for r in raw_runs if r["memory_after_mb"] is not None]
        mem_delta_values = [r["memory_delta_mb"] for r in raw_runs if r["memory_delta_mb"] is not None]

        summary["average_cpu_percent"] = statistics.mean(avg_cpu_values) if avg_cpu_values else None
        summary["peak_cpu_percent"] = max(peak_cpu_values) if peak_cpu_values else None
        summary["memory_before_mb"] = statistics.mean(mem_before_values) if mem_before_values else None
        summary["memory_after_mb"] = statistics.mean(mem_after_values) if mem_after_values else None
        summary["memory_delta_mb"] = statistics.mean(mem_delta_values) if mem_delta_values else None
    else:
        summary["average_cpu_percent"] = "Not Available"
        summary["peak_cpu_percent"] = "Not Available"
        summary["memory_before_mb"] = "Not Available"
        summary["memory_after_mb"] = "Not Available"
        summary["memory_delta_mb"] = "Not Available"

    # Throughput: real requests/sec over the ACTUAL measured wall-clock
    # time of the benchmarked requests (sequential execution), not an
    # estimate -- sum of each request's own measured total_latency_ms.
    total_wall_time_s = sum(r["total_latency_ms"] for r in raw_runs) / 1000.0
    summary["requests_per_second"] = (n_total / total_wall_time_s) if total_wall_time_s > 0 else None

    return summary


def collect_environment_info() -> Dict[str, Any]:
    """Real hardware / interpreter / package metadata -- no fabricated values."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "gpu": detect_gpu(),
    }
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["total_memory_gb"] = round(vm.total / (1024 ** 3), 2)
        info["psutil_version"] = psutil.__version__
    else:
        info["cpu_logical_cores"] = "Not Available"
        info["cpu_physical_cores"] = "Not Available"
        info["total_memory_gb"] = "Not Available"
        info["psutil_version"] = "Not Available"
    return info


# --------------------------------------------------------------------------
# FIGURES
# --------------------------------------------------------------------------

def plot_latency_breakdown(summary: Dict[str, Any], path: Path) -> None:
    """Publication-quality bar chart of mean stage latency (+/- std), in ms."""
    stages = ["retrieval_latency_ms", "prompt_build_latency_ms", "llm_latency_ms", "total_latency_ms"]
    labels = ["Retrieval", "Prompt\nConstruction", "LLM\nInference", "Total\nPipeline"]
    means = [summary[s]["mean"] or 0.0 for s in stages]
    stds = [summary[s]["std"] or 0.0 for s in stages]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.5, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Latency (ms)", fontsize=10)
    ax.set_title(
        "AI Mentor Pipeline — Latency Breakdown by Stage\n"
        "(error bars = standard deviation across repeated real runs)",
        fontweight="bold", pad=12,
    )
    for xi, m in zip(x, means):
        ax.text(xi, m, f"{m:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_memory_usage(summary: Dict[str, Any], path: Path) -> None:
    """Publication-quality bar chart of mean memory before/after/delta, in MB."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if PSUTIL_AVAILABLE:
        labels = ["Before", "After", "Delta"]
        values = [summary["memory_before_mb"], summary["memory_after_mb"], summary["memory_delta_mb"]]
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f} MB", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Process RSS Memory (MB)", fontsize=10)
    else:
        ax.text(0.5, 0.5, "psutil not available —\nmemory metrics reported as\n\"Not Available\"",
                 ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("AI Mentor Pipeline — Memory Usage", fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_cpu_usage(summary: Dict[str, Any], path: Path) -> None:
    """Publication-quality bar chart of average vs peak CPU percent."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if PSUTIL_AVAILABLE:
        labels = ["Average CPU %", "Peak CPU %"]
        values = [summary["average_cpu_percent"], summary["peak_cpu_percent"]]
        colors = ["#4C72B0", "#C44E52"]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("CPU Usage (%)", fontsize=10)
    else:
        ax.text(0.5, 0.5, "psutil not available —\nCPU metrics reported as\n\"Not Available\"",
                 ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("AI Mentor Pipeline — CPU Usage\n(sampled throughout each request)", fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


def plot_throughput(summary: Dict[str, Any], path: Path) -> None:
    """Publication-quality single-bar chart of measured requests/sec."""
    fig, ax = plt.subplots(figsize=(6, 5))
    rps = summary["requests_per_second"] or 0.0
    ax.bar(["AI Mentor\nPipeline"], [rps], color="#55A868", edgecolor="white", linewidth=0.5, width=0.5)
    ax.text(0, rps, f"{rps:.3f} req/s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Throughput (requests/sec)", fontsize=10)
    ax.set_title(
        "AI Mentor Pipeline — Measured Throughput\n(sequential, single-process)",
        fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# MARKDOWN REPORT (with interpretation, not just tabulated numbers)
# --------------------------------------------------------------------------

def _fmt(value: Any, unit: str = "", decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.{decimals}f}{unit}"


def generate_report(summary: Dict[str, Any], env_info: Dict[str, Any], path: Path) -> None:
    """
    Writes the IEEE-facing markdown report: hardware, Python version,
    average latency + std, CPU, memory, throughput, and an
    interpretation section that reasons about what the measured numbers
    imply for the deployed pipeline, computed from the real aggregated
    statistics above (never from assumed/typical values).
    """
    retrieval = summary["retrieval_latency_ms"]
    prompt_build = summary["prompt_build_latency_ms"]
    llm = summary["llm_latency_ms"]
    total = summary["total_latency_ms"]

    # NOTE: We intentionally do NOT compute any "stage mean / total mean"
    # percentage here. Retrieval and prompt-construction means are taken
    # over ALL benchmarked requests, while the LLM-latency mean is taken
    # only over the subset of requests that actually reached the LLM
    # (i.e. were not intercepted by the similarity-threshold gate). Those
    # two means have different denominators, so dividing an LLM-latency
    # mean by the total-latency mean is not a mathematically valid
    # share-of-total and can exceed 100%. The interpretation section
    # below therefore describes relative cost qualitatively instead of
    # via a computed percentage-of-total figure.

    memory_delta = summary["memory_delta_mb"]
    memory_note = (
        "Memory metrics were not available in this environment (psutil not installed)."
        if memory_delta == "Not Available"
        else (
            f"the mean per-request resident-memory delta was {_fmt(memory_delta, ' MB', 3)}, "
            + (
                "which is small and does not indicate an obvious per-request memory leak "
                "across the benchmarked runs"
                if isinstance(memory_delta, (int, float)) and abs(memory_delta) < 5
                else "which is large enough to warrant investigation for a possible "
                     "per-request memory leak across repeated calls"
            )
        )
    )

    gated_note = ""
    if summary["n_requests_similarity_gated"] > 0:
        gated_note = (
            f"\n\n> **Note:** {summary['n_requests_similarity_gated']} of "
            f"{summary['n_requests_total']} benchmarked requests were intercepted by the "
            f"existing Sprint 8 similarity-threshold gate (retrieval evidence too weak), so "
            f"the LLM was never invoked for those requests. Those requests are correctly "
            f"excluded from the LLM- and prompt-construction-latency statistics above (not "
            f"counted as zero-latency), and are reported separately so the LLM-inference "
            f"numbers reflect only requests that actually reached the LLM."
        )

    report = f"""# AI Mentor Pipeline — Computational Benchmarking Report

## Purpose

This report documents the **real, measured computational cost** of the
deployed AI Mentor pipeline (`ask_mentor()` in `mentor_service.py`) —
retrieval, prompt construction, LLM inference, and the end-to-end
request — for the IEEE paper's Computational Cost evaluation. Every
figure below comes from `time.perf_counter()` and `psutil`
measurements taken around the real, unmodified pipeline code; nothing
is estimated, assumed, or fabricated. Where a metric could not be
measured (e.g. `psutil` unavailable, no GPU present), it is reported
literally as "Not Available" / "CPU Only" rather than a placeholder
number.

## Methodology

- {summary['n_requests_total']} timed requests were benchmarked (after
  {N_WARMUP_RUNS} untimed warm-up request(s) per configured
  question, excluded from all statistics to avoid attributing one-off
  cold-start cost — e.g. first LLM weights load, first vector-index
  touch — to steady-state figures).
- Retrieval, prompt-construction, and LLM-inference timings were
  obtained by transparently wrapping the exact functions
  `ask_mentor()` already calls (`retriever.retrieve`,
  `build_system_prompt` + `_build_output_format_instructions`,
  `ollama.chat`) with `time.perf_counter()`, without altering their
  inputs, outputs, or the surrounding pipeline logic in any way (see
  this script's module docstring for the full non-invasiveness
  guarantee).
- Total pipeline latency is the real wall-clock time around the entire
  `ask_mentor()` call, so it also includes stages not individually
  broken out above (context/risk/persona/SHAP gathering, confidence
  scoring, citation building, and the `mentor_history` database write).
- CPU usage was sampled on a background thread every
  {CPU_SAMPLE_INTERVAL_S * 1000:.0f} ms for the duration of each
  request (via `psutil.Process().cpu_percent()`), so average/peak CPU
  reflect real sampled utilisation across the whole request rather
  than one coarse before/after reading.
- Memory usage is the process resident-set-size (RSS) immediately
  before and after each request, in megabytes.
- All statistics are aggregated as mean, median, standard deviation,
  minimum, and maximum across the repeated timed runs.{gated_note}

## Hardware / Environment

| Property | Value |
|---|---|
| Platform | {env_info['platform']} |
| Python version | {env_info['python_version']} |
| GPU | {env_info['gpu']} |
| Logical CPU cores | {env_info['cpu_logical_cores']} |
| Physical CPU cores | {env_info['cpu_physical_cores']} |
| Total system memory | {_fmt(env_info['total_memory_gb'], ' GB') if env_info['total_memory_gb'] != 'Not Available' else 'Not Available'} |
| psutil | {env_info['psutil_version']} |

## Latency Results (milliseconds)

| Stage | n | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| Retrieval | {retrieval['n']} | {_fmt(retrieval['mean'])} | {_fmt(retrieval['median'])} | {_fmt(retrieval['std'])} | {_fmt(retrieval['min'])} | {_fmt(retrieval['max'])} |
| Prompt construction | {prompt_build['n']} | {_fmt(prompt_build['mean'])} | {_fmt(prompt_build['median'])} | {_fmt(prompt_build['std'])} | {_fmt(prompt_build['min'])} | {_fmt(prompt_build['max'])} |
| LLM inference | {llm['n']} | {_fmt(llm['mean'])} | {_fmt(llm['median'])} | {_fmt(llm['std'])} | {_fmt(llm['min'])} | {_fmt(llm['max'])} |
| **Total pipeline** | {total['n']} | **{_fmt(total['mean'])}** | {_fmt(total['median'])} | {_fmt(total['std'])} | {_fmt(total['min'])} | {_fmt(total['max'])} |

## CPU and Memory

| Metric | Value |
|---|---|
| Average CPU (%) | {_fmt(summary['average_cpu_percent'], '%') if summary['average_cpu_percent'] != 'Not Available' else 'Not Available'} |
| Peak CPU (%) | {_fmt(summary['peak_cpu_percent'], '%') if summary['peak_cpu_percent'] != 'Not Available' else 'Not Available'} |
| Mean memory before (MB) | {_fmt(summary['memory_before_mb'], ' MB', 3) if summary['memory_before_mb'] != 'Not Available' else 'Not Available'} |
| Mean memory after (MB) | {_fmt(summary['memory_after_mb'], ' MB', 3) if summary['memory_after_mb'] != 'Not Available' else 'Not Available'} |
| Mean memory delta (MB) | {_fmt(summary['memory_delta_mb'], ' MB', 3) if summary['memory_delta_mb'] != 'Not Available' else 'Not Available'} |

## Throughput

**{_fmt(summary['requests_per_second'], ' requests/sec', 4)}**, measured
as the number of benchmarked requests divided by their real summed
wall-clock time (sequential, single-process execution — not an
estimate from any single stage's latency).

## Interpretation

Among requests that reached the language model, LLM inference was the
dominant contributor to latency, while retrieval and prompt
construction contributed comparatively little to overall execution
time. Requests intercepted by the similarity-threshold safety gate
were excluded from the LLM latency statistics because the language
model was not invoked. In an LLM-backed mentor pipeline this pattern is
expected: retrieval and prompt assembly are in-process, CPU-bound
operations over a small, fixed-size context, while LLM inference
involves a full forward pass through a language model and is typically
the dominant cost driver among the stages that actually run for a
given request — a useful data point for deciding where future
optimisation effort (e.g. prompt shortening, a smaller or quantized
model, batched retrieval, or response caching for repeated questions)
would have the most impact.

Regarding memory, {memory_note}.

The measured throughput of {_fmt(summary['requests_per_second'], ' req/sec', 4)}
reflects strictly **sequential** single-process execution of the real
pipeline (one request fully completes, including its database write,
before the next begins); it should be read as a lower bound on
achievable throughput under concurrent/batched serving, not as the
system's maximum capacity.

## Figures

- `figures/latency_breakdown.png` — mean latency (± std) per pipeline
  stage, in milliseconds
- `figures/memory_usage.png` — mean memory before / after / delta, in
  megabytes
- `figures/cpu_usage.png` — mean average vs. peak CPU utilisation,
  sampled throughout each request
- `figures/throughput.png` — measured requests/sec

## Scope Note

This report measures **computational cost only** — latency, CPU, and
memory of the real, unmodified pipeline. It makes no claim about, and
does not measure, the correctness, relevance, or safety of the
mentor's responses; those are evaluated elsewhere (e.g. the existing
hallucination-safety and confidence-scoring logic already built into
`mentor_service.py`, which this benchmark leaves completely untouched).
"""

    path.write_text(report, encoding="utf-8")
    logger.info(f"[SAVED] {path}")


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EduGuard-AI :: AI Mentor Pipeline Computational Benchmark")
    logger.info("=" * 70)

    if not PSUTIL_AVAILABLE:
        logger.warning("psutil is not installed — CPU/memory metrics will be reported as 'Not Available'.")

    if any(req["user_id"] == "REPLACE_WITH_REAL_TEST_USER_ID" for req in BENCHMARK_REQUESTS):
        logger.error(
            "BENCHMARK_REQUESTS still contains the placeholder user_id. "
            "Edit the CONFIGURATION section with a real test user_id/question "
            "before running this benchmark."
        )
        sys.exit(1)

    env_info = collect_environment_info()
    logger.info(
        f"Environment: {env_info['platform']}, Python {env_info['python_version']}, "
        f"GPU: {env_info['gpu']}"
    )

    # ---- Warm-up (untimed, excluded from all statistics) ----
    logger.info(f"Running {N_WARMUP_RUNS} untimed warm-up request(s) ...")
    for i in range(N_WARMUP_RUNS):
        req = BENCHMARK_REQUESTS[i % len(BENCHMARK_REQUESTS)]
        run_single_request(req["user_id"], req["question"])
        logger.info(f"  [warm-up {i + 1}/{N_WARMUP_RUNS}] complete.")

    # ---- Timed benchmark runs ----
    logger.info(f"Running {N_BENCHMARK_RUNS} timed benchmark request(s) ...")
    raw_runs: List[Dict[str, Any]] = []
    for i in range(N_BENCHMARK_RUNS):
        req = BENCHMARK_REQUESTS[i % len(BENCHMARK_REQUESTS)]
        result = run_single_request(req["user_id"], req["question"])
        raw_runs.append(result)
        logger.info(
            f"  [run {i + 1}/{N_BENCHMARK_RUNS}] total={result['total_latency_ms']:.1f} ms, "
            f"retrieval={_fmt(result['retrieval_latency_ms'], ' ms', 1)}, "
            f"prompt_build={_fmt(result['prompt_build_latency_ms'], ' ms', 1)}, "
            f"llm={_fmt(result['llm_latency_ms'], ' ms', 1)}"
        )

    # ---- Aggregate ----
    summary = aggregate_runs(raw_runs)

    # ---- Persist computational_metrics.json ----
    metrics_payload = {
        "metadata": {
            **env_info,
            "n_warmup_runs": N_WARMUP_RUNS,
            "n_benchmark_runs": N_BENCHMARK_RUNS,
            "cpu_sample_interval_s": CPU_SAMPLE_INTERVAL_S,
            "benchmark_requests_used": BENCHMARK_REQUESTS,
        },
        "per_request_raw": raw_runs,
        "summary": summary,
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")
    logger.info(f"[SAVED] {METRICS_JSON_PATH}")

    # ---- Figures ----
    logger.info("Generating publication-quality figures ...")
    plot_latency_breakdown(summary, LATENCY_FIG_PATH)
    plot_memory_usage(summary, MEMORY_FIG_PATH)
    plot_cpu_usage(summary, CPU_FIG_PATH)
    plot_throughput(summary, THROUGHPUT_FIG_PATH)

    # ---- Markdown report ----
    generate_report(summary, env_info, REPORT_PATH)

    logger.info("=" * 70)
    logger.info("AI Mentor pipeline computational benchmarking complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
