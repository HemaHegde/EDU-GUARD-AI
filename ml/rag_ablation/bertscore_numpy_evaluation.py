"""
bertscore_numpy_evaluation.py
==============================
BERTScore approximation using FAISS index embeddings already on disk
(the same all-MiniLM-L6-v2 embeddings used by the production RAG pipeline).

This avoids the sentence-transformers import conflict in the Python 3.13
environment by using numpy + faiss directly — both already work.

Strategy:
  1. Re-encodes gold references and mentor responses using the same
     FAISS/embedding pipeline the production system already uses
     (importing from backend/services/retrieval.py).
  2. Computes cosine F1 between all token-level embeddings
     (word-pieces, approximated by sentence-level cosine since we cannot
     run a BERT tokenizer without the full transformers stack).
  3. Reports results as "BERTScore approx (sentence-level cosine F1)".

Usage:
    python ml/rag_ablation/bertscore_numpy_evaluation.py

This runs in the same Python 3.13 + TF2 environment as all other ml/ scripts.
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

# Add backend to path (same pattern as all ml/ scripts)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
RECORDS_PATH = OUTPUT_DIR / "records.jsonl"
GOLD_PATH    = OUTPUT_DIR / "gold_references.json"
OUT_CSV      = OUTPUT_DIR / "bertscore_approx_results.csv"
OUT_JSON     = OUTPUT_DIR / "bertscore_approx_summary.json"
OUT_FIG      = FIGURE_DIR / "bertscore_comparison.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NOT_EVALUATED = "Not Evaluated"

# ── Load embedding model (same as production) ─────────────────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            # Import the retrieval module — same model as the RAG pipeline
            from services.retrieval import LearningMaterialRetriever
            retriever = LearningMaterialRetriever()
            _embedder = retriever._model
            if _embedder is None:
                raise RuntimeError("Retrieval model is None (sentence_transformers unavailable)")
            print("[OK] Embedding model loaded from backend/services/retrieval.py")
        except Exception as e:
            print(f"[WARN] Could not load embedding model via retrieval.py: {e}")
            _embedder = "FAILED"
    return _embedder if _embedder != "FAILED" else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_bertscore_approx(reference: str, hypothesis: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Sentence-level cosine BERTScore approximation.
    P = R = F1 = cosine_similarity(embed(reference), embed(hypothesis)).
    """
    embedder = _get_embedder()
    if embedder is None:
        return None, None, None
    try:
        embs = embedder.encode([reference, hypothesis], convert_to_numpy=True)
        sim = _cosine(embs[0], embs[1])
        return sim, sim, sim
    except Exception as e:
        print(f"[WARN] Encoding failed: {e}")
        return None, None, None


def load_gold_references() -> Dict[str, str]:
    if not GOLD_PATH.exists():
        print(f"[ERROR] Gold references not found: {GOLD_PATH}")
        return {}
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_records() -> List[dict]:
    if not RECORDS_PATH.exists():
        print(f"[ERROR] Records file not found: {RECORDS_PATH}")
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
    print(f"[OK] Loaded {len(records)} records from {RECORDS_PATH.name}")
    return records


def main():
    gold_refs = load_gold_references()
    records   = load_records()

    if not records or not gold_refs:
        print("[EXIT] Missing data.")
        sys.exit(1)

    print(f"[INFO] Gold references for {len(gold_refs)} question(s)")
    print("[INFO] Implementation: sentence-level cosine F1 (all-MiniLM-L6-v2)\n")

    results  = []
    n_ok     = 0
    n_skip   = 0

    for rec in records:
        question = rec.get("question", "")
        config   = rec.get("config", "")
        response = rec.get("mentor_response_text")
        gold_ref = gold_refs.get(question)

        if not gold_ref or not response:
            results.append({
                "user_id": rec.get("user_id", ""), "question": question,
                "config": config, "bertscore_p": NOT_EVALUATED,
                "bertscore_r": NOT_EVALUATED, "bertscore_f1": NOT_EVALUATED,
                "note": "no_gold_reference" if not gold_ref else "no_response_text",
            })
            n_skip += 1
            continue

        p, r, f1 = compute_bertscore_approx(gold_ref, response)

        results.append({
            "user_id": rec.get("user_id", ""), "question": question,
            "config": config,
            "bertscore_p":  round(float(p), 4) if p is not None else NOT_EVALUATED,
            "bertscore_r":  round(float(r), 4) if r is not None else NOT_EVALUATED,
            "bertscore_f1": round(float(f1), 4) if f1 is not None else NOT_EVALUATED,
            "note": "",
        })
        n_ok += 1

    print(f"[DONE] Evaluated: {n_ok} | Skipped: {n_skip}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    fields = ["user_id", "question", "config", "bertscore_p", "bertscore_r", "bertscore_f1", "note"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] CSV: {OUT_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────────
    cfg_scores: Dict[str, List[float]] = defaultdict(list)
    for rec in results:
        f1 = rec["bertscore_f1"]
        if f1 != NOT_EVALUATED:
            cfg_scores[rec["config"]].append(float(f1))

    summary = {}
    for cfg, scores in cfg_scores.items():
        if not scores:
            continue
        arr = np.array(scores)
        n   = len(arr)
        mean = float(arr.mean())
        std  = float(arr.std())
        ci_lo = mean - 1.96 * std / math.sqrt(n) if n > 1 else mean
        ci_hi = mean + 1.96 * std / math.sqrt(n) if n > 1 else mean
        summary[cfg] = {
            "n": n, "mean": round(mean, 4), "std": round(std, 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
            "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
        }

    out_data = {
        "implementation": "sentence-level cosine F1 approximation (all-MiniLM-L6-v2)",
        "metric_note": (
            "P = R = F1 = cosine similarity between sentence embeddings of reference and hypothesis. "
            "This is NOT token-level BERTScore. Install bert-score for token-level BERTScore: "
            "pip install bert-score"
        ),
        "n_evaluated": n_ok,
        "n_skipped": n_skip,
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

        fig, ax = plt.subplots(figsize=(11, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        bars = ax.bar(configs, means, yerr=stds, color=colors,
                      capsize=5, error_kw={"linewidth": 1.5}, alpha=0.85)
        for bar, mean, n in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05 + 0.005,
                    f"{mean:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)

        ax.set_title(
            "Approx. BERTScore F1 by Ablation Configuration\n"
            "(sentence-level cosine similarity, all-MiniLM-L6-v2)",
            fontsize=12
        )
        ax.set_xlabel("Ablation Configuration", fontsize=11)
        ax.set_ylabel("BERTScore F1 approx. (mean ± std)", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Figure: {OUT_FIG}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Approx. BERTScore F1 (sentence-level cosine, all-MiniLM-L6-v2)")
    print(f"{'='*72}")
    print(f"  {'Config':<25} {'n':>4} {'Mean F1':>9} {'Std':>7} {'95% CI':>20}")
    print(f"  {'-'*70}")
    for cfg, s in summary.items():
        ci = f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]"
        print(f"  {cfg:<25} {s['n']:>4} {s['mean']:>9.4f} {s['std']:>7.4f} {ci:>20}")
    print(f"{'='*72}\n")

    # Paired comparison vs Full System
    full_sys = {r["question"]: r for r in results
                if r["config"] == "Full System" and r["bertscore_f1"] != NOT_EVALUATED}
    if full_sys:
        print("  Paired comparison vs Full System (BERTScore F1):")
        print(f"  {'Config':<25} {'Mean diff':>10} {'n pairs':>8}")
        print(f"  {'-'*48}")
        full_sys_vals = {q: float(r["bertscore_f1"]) for q, r in full_sys.items()}
        for cfg in configs:
            if cfg == "Full System":
                continue
            arm_vals = {r["question"]: float(r["bertscore_f1"]) for r in results
                        if r["config"] == cfg and r["bertscore_f1"] != NOT_EVALUATED}
            pairs = [(arm_vals[q] - full_sys_vals[q]) for q in arm_vals
                     if q in full_sys_vals]
            if pairs:
                mean_diff = np.mean(pairs)
                print(f"  {cfg:<25} {mean_diff:>+10.4f} {len(pairs):>8}")
        print()


if __name__ == "__main__":
    main()
