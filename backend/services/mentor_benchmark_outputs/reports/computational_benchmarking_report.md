# AI Mentor Pipeline — Computational Benchmarking Report

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

- 5 timed requests were benchmarked (after
  1 untimed warm-up request(s) per configured
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
  50 ms for the duration of each
  request (via `psutil.Process().cpu_percent()`), so average/peak CPU
  reflect real sampled utilisation across the whole request rather
  than one coarse before/after reading.
- Memory usage is the process resident-set-size (RSS) immediately
  before and after each request, in megabytes.
- All statistics are aggregated as mean, median, standard deviation,
  minimum, and maximum across the repeated timed runs.

> **Note:** 2 of 5 benchmarked requests were intercepted by the existing Sprint 8 similarity-threshold gate (retrieval evidence too weak), so the LLM was never invoked for those requests. Those requests are correctly excluded from the LLM- and prompt-construction-latency statistics above (not counted as zero-latency), and are reported separately so the LLM-inference numbers reflect only requests that actually reached the LLM.

## Hardware / Environment

| Property | Value |
|---|---|
| Platform | Windows-10-10.0.26200-SP0 |
| Python version | 3.11.5 |
| GPU | CPU Only |
| Logical CPU cores | 22 |
| Physical CPU cores | 16 |
| Total system memory | 31.61 GB |
| psutil | 7.2.2 |

## Latency Results (milliseconds)

| Stage | n | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| Retrieval | 5 | 95.85 | 47.19 | 120.64 | 35.15 | 311.32 |
| Prompt construction | 3 | 1.60 | 1.52 | 0.20 | 1.45 | 1.83 |
| LLM inference | 3 | 41894.83 | 41943.05 | 307.69 | 41565.88 | 42175.56 |
| **Total pipeline** | 5 | **26161.05** | 42507.10 | 23099.86 | 714.05 | 43562.43 |

## CPU and Memory

| Metric | Value |
|---|---|
| Average CPU (%) | 191.50% |
| Peak CPU (%) | 1868.20% |
| Mean memory before (MB) | 806.695 MB |
| Mean memory after (MB) | 807.138 MB |
| Mean memory delta (MB) | 0.444 MB |

## Throughput

**0.0382 requests/sec**, measured
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

Regarding memory, the mean per-request resident-memory delta was 0.444 MB, which is small and does not indicate an obvious per-request memory leak across the benchmarked runs.

The measured throughput of 0.0382 req/sec
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
