"""
bertscore_evaluation.py
========================
Computes BERTScore (Precision, Recall, F1) for all RAG ablation
mentor responses against the gold reference answers already stored in
ml/rag_ablation/outputs/gold_references.json.

This script reads:
  ml/rag_ablation/outputs/quality_records.jsonl   <- generated responses
  ml/rag_ablation/outputs/gold_references.json    <- gold references

It writes:
  ml/rag_ablation/outputs/bertscore_results.csv   <- per-record P/R/F1
  ml/rag_ablation/outputs/bertscore_summary.json  <- mean/std per config
  ml/rag_ablation/figures/bertscore_comparison.png <- comparison figure

BERTScore implementation:
  - Attempts to import the `bert_score` package (pip install bert-score).
  - If unavailable, falls back to a cosine-similarity approximation using
    sentence-transformers all-MiniLM-L6-v2 embeddings (token-level cosine
    F1), which is already installed in this project.

No student data is fabricated. Every row reads `Not Evaluated` when
a gold reference is unavailable for that question.

Usage:
    cd <project root>
    python ml/rag_ablation/bertscore_evaluation.py

    # Or with real bert-score package installed:
    pip install bert-score
    python ml/rag_ablation/bertscore_evaluation.py
"""

import json
import os
import sys
import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
OUTPUT_DIR     = SCRIPT_DIR / "outputs"
FIGURE_DIR     = SCRIPT_DIR / "figures"
RECORDS_PATH   = OUTPUT_DIR / "records.jsonl"
GOLD_PATH      = OUTPUT_DIR / "gold_references.json"
OUT_CSV        = OUTPUT_DIR / "bertscore_results.csv"
OUT_JSON       = OUTPUT_DIR / "bertscore_summary.json"
OUT_FIG        = FIGURE_DIR / "bertscore_comparison.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NOT_EVALUATED = "Not Evaluated"

# ── Attempt real bert-score import ─────────────────────────────────────────────
USING_REAL_BERTSCORE = False
_bert_scorer = None

try:
    from bert_score import score as bert_score_fn  # type: ignore
    USING_REAL_BERTSCORE = True
    print("[OK] bert-score package imported successfully.")
except ImportError:
    print("[WARN] bert-score package not installed. Falling back to "
          "sentence-transformers cosine F1 approximation.")
    print("       To use real BERTScore: pip install bert-score")

# ── Sentence-Transformers fallback ─────────────────────────────────────────────
_ST_MODEL = None

def _load_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            import os as _os
            _os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            _os.environ.setdefault("KERAS_BACKEND", "jax")
            from sentence_transformers import SentenceTransformer  # type: ignore
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("[OK] sentence-transformers model loaded for BERTScore fallback.")
        except Exception as e:
            print(f"[ERROR] Could not load sentence-transformers: {e}")
            # Try numpy-only cosine using cached embeddings if available
    return _ST_MODEL


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _bertscore_fallback(reference: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    Approximate BERTScore using sentence-level cosine similarity.
    Returns (precision, recall, f1) where all three equal the cosine
    similarity between the reference and hypothesis embeddings.

    NOTE: This is a sentence-level approximation, not token-level BERTScore.
    It is reported as an approximation in all outputs. Real BERTScore
    requires the `bert-score` package.
    """
    model = _load_st_model()
    if model is None:
        return None, None, None
    try:
        embs = model.encode([reference, hypothesis], convert_to_numpy=True)
        sim = _cosine(embs[0], embs[1])
        # For sentence-level: P = R = F = sim (same representation)
        return sim, sim, sim
    except Exception as e:
        print(f"[WARN] Fallback BERTScore failed: {e}")
        return None, None, None


def compute_bertscore(reference: str, hypothesis: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute BERTScore P/R/F1.
    Uses real bert-score if available, else sentence-transformers fallback.
    """
    if not reference or not hypothesis:
        return None, None, None

    if USING_REAL_BERTSCORE:
        try:
            P, R, F = bert_score_fn(
                [hypothesis], [reference],
                lang="en",
                model_type="distilbert-base-uncased",
                verbose=False,
            )
            return float(P[0]), float(R[0]), float(F[0])
        except Exception as e:
            print(f"[WARN] bert-score computation failed: {e}. Falling back.")

    return _bertscore_fallback(reference, hypothesis)


# ── Load data ──────────────────────────────────────────────────────────────────

def load_gold_references() -> Dict[str, str]:
    if not GOLD_PATH.exists():
        print(f"[ERROR] Gold references file not found: {GOLD_PATH}")
        return {}
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_quality_records() -> List[dict]:
    if not RECORDS_PATH.exists():
        print(f"[ERROR] Quality records file not found: {RECORDS_PATH}")
        return []
    records = []
    with open(RECORDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"[OK] Loaded {len(records)} quality records from {RECORDS_PATH.name}")
    return records


# ── Main evaluation ────────────────────────────────────────────────────────────

def main():
    gold_refs = load_gold_references()
    records   = load_quality_records()

    if not records:
        print("[EXIT] No records to evaluate.")
        sys.exit(1)

    if not gold_refs:
        print("[EXIT] No gold references found.")
        sys.exit(1)

    print(f"\n[INFO] Gold references available for {len(gold_refs)} question(s).")
    print(f"[INFO] BERTScore implementation: "
          f"{'real bert-score (distilbert-base-uncased)' if USING_REAL_BERTSCORE else 'sentence-transformers cosine F1 approximation (all-MiniLM-L6-v2)'}\n")

    results = []
    skipped = 0

    for rec in records:
        question = rec.get("question", "")
        config   = rec.get("config", "")
        response = rec.get("mentor_response_text")

        gold_ref = gold_refs.get(question)

        if not gold_ref or not response:
            results.append({
                "user_id":        rec.get("user_id", ""),
                "question":       question,
                "config":         config,
                "bertscore_p":    NOT_EVALUATED,
                "bertscore_r":    NOT_EVALUATED,
                "bertscore_f1":   NOT_EVALUATED,
                "note":           "no_gold_reference" if not gold_ref else "no_response_text",
                "implementation": "real_bertscore" if USING_REAL_BERTSCORE else "cosine_approximation",
            })
            skipped += 1
            continue

        p, r, f1 = compute_bertscore(gold_ref, response)

        results.append({
            "user_id":        rec.get("user_id", ""),
            "question":       question,
            "config":         config,
            "bertscore_p":    round(float(p), 4) if p is not None else NOT_EVALUATED,
            "bertscore_r":    round(float(r), 4) if r is not None else NOT_EVALUATED,
            "bertscore_f1":   round(float(f1), 4) if f1 is not None else NOT_EVALUATED,
            "note":           "",
            "implementation": "real_bertscore" if USING_REAL_BERTSCORE else "cosine_approximation",
        })

    evaluated = len(results) - skipped
    print(f"[DONE] Evaluated: {evaluated} records | Skipped (no gold/response): {skipped}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fields = ["user_id", "question", "config",
              "bertscore_p", "bertscore_r", "bertscore_f1", "note", "implementation"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] Per-record CSV: {OUT_CSV}")

    # ── Aggregate by configuration ─────────────────────────────────────────────
    config_scores: Dict[str, List[float]] = defaultdict(list)
    for rec in results:
        f1 = rec["bertscore_f1"]
        if f1 != NOT_EVALUATED:
            try:
                config_scores[rec["config"]].append(float(f1))
            except (TypeError, ValueError):
                pass

    summary = {}
    for cfg, scores in config_scores.items():
        if not scores:
            continue
        arr = np.array(scores)
        n = len(arr)
        mean = float(arr.mean())
        std  = float(arr.std())
        ci95_lo = mean - 1.96 * std / math.sqrt(n) if n > 1 else mean
        ci95_hi = mean + 1.96 * std / math.sqrt(n) if n > 1 else mean
        summary[cfg] = {
            "n":      n,
            "mean":   round(mean, 4),
            "std":    round(std, 4),
            "min":    round(float(arr.min()), 4),
            "max":    round(float(arr.max()), 4),
            "ci95_lo": round(ci95_lo, 4),
            "ci95_hi": round(ci95_hi, 4),
        }

    out_data = {
        "implementation": "real_bertscore (distilbert-base-uncased)" if USING_REAL_BERTSCORE
                          else "cosine_approximation (all-MiniLM-L6-v2 sentence embeddings)",
        "metric_note": ("Real BERTScore token-level P/R/F1." if USING_REAL_BERTSCORE
                        else "Approximation: sentence-level cosine similarity. "
                             "Install bert-score for token-level BERTScore."),
        "n_evaluated": evaluated,
        "n_skipped":   skipped,
        "per_config_bertscore_f1": summary,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"[OK] Summary JSON: {OUT_JSON}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    if summary:
        configs = list(summary.keys())
        means   = [summary[c]["mean"] for c in configs]
        stds    = [summary[c]["std"]  for c in configs]
        ns      = [summary[c]["n"]    for c in configs]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        bars = ax.bar(configs, means, yerr=stds, color=colors,
                      capsize=5, error_kw={"linewidth": 1.5}, alpha=0.85)

        for bar, mean, n in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f"{mean:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)

        impl_label = ("Real BERTScore (distilbert-base-uncased)"
                      if USING_REAL_BERTSCORE
                      else "Approx. BERTScore (all-MiniLM-L6-v2 cosine, n/a token-level)")
        ax.set_title(f"BERTScore F1 by Ablation Configuration\n{impl_label}", fontsize=12)
        ax.set_xlabel("Ablation Configuration", fontsize=11)
        ax.set_ylabel("BERTScore F1 (mean ± std)", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Figure: {OUT_FIG}")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BERTScore F1 Summary by Configuration")
    print(f"  Implementation: {'real bert-score' if USING_REAL_BERTSCORE else 'cosine approximation (fallback)'}")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'n':>4} {'Mean F1':>9} {'Std':>7} {'95% CI':>18}")
    print(f"  {'-'*68}")
    for cfg, s in summary.items():
        ci = f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]"
        print(f"  {cfg:<25} {s['n']:>4} {s['mean']:>9.4f} {s['std']:>7.4f} {ci:>18}")
    print(f"{'='*70}")
    print(f"\n  Outputs:")
    print(f"    {OUT_CSV}")
    print(f"    {OUT_JSON}")
    print(f"    {OUT_FIG}")
    if not USING_REAL_BERTSCORE:
        print(f"\n  NOTE: Results are cosine-similarity approximations.")
        print(f"        Install the real bert-score package for token-level BERTScore:")
        print(f"        pip install bert-score")
    print()


if __name__ == "__main__":
    main()
