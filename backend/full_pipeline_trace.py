"""
Full End-to-End Pipeline Trace
==============================
Traces ALL stages of the pipeline for the 3 evaluation scenarios
WITHOUT hitting the live API — uses the actual service modules directly.

Run from: d:\\EDU AI\\edu-ai\\backend
Usage: python full_pipeline_trace.py

Shows:
  Seed Profile -> DB Values -> Risk Service -> Persona Service ->
  Context Builder -> Priority Focus (Reasoning) -> Prompt Builder ->
  Final Persona / Risk Level / Response Theme
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Load ML models once ──────────────────────────────────────────────────────
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
KMEANS_PATH  = os.path.join(BASE, "../ml/persona_kmeans.pkl")
SCALER_PATH  = os.path.join(BASE, "../ml/persona_scaler.pkl")
XGBOOST_PATH = os.path.join(BASE, "../ml/academic_risk_xgboost.pkl")

persona_model  = joblib.load(KMEANS_PATH)
persona_scaler = joblib.load(SCALER_PATH)
xgb_model      = joblib.load(XGBOOST_PATH)

PERSONA_MAP = {
    0: "Silent Isolator",
    1: "Burnout Pattern",
    2: "Anxiety-Spike Learner",
    3: "Passive Watcher",
    4: "Consistent Learner",
    5: "Last-Minute Survivor",
}

# ─── Seed profiles (from seed_service.py) ────────────────────────────────────
from services.seed_service import DEMO_PROFILES

EVAL_SCENARIOS = {
    "LOW (S-LR-001)": {
        "seed_id":          "S-LR-001",
        "profile_key":      "consistent",
        "expected_persona": "Consistent Learner",
        "expected_risk":    "Low",
    },
    "MODERATE (S-MR-002)": {
        "seed_id":          "S-MR-002",
        "profile_key":      "last_minute",
        "expected_persona": "Last-Minute Survivor",
        "expected_risk":    "High or Medium",
    },
    "HIGH (S-HR-003)": {
        "seed_id":          "S-HR-003",
        "profile_key":      "burnout",
        "expected_persona": "Burnout Pattern",
        "expected_risk":    "High",
    },
}


def predict_risk(features):
    """Run features through XGBoost risk model."""
    df = pd.DataFrame([{
        "total_clicks":            features["total_clicks"],
        "avg_score":               features["avg_score"],
        "active_days":             features["active_days"],
        "engagement_variability":  features["engagement_variability"],
        "inactivity_days":         features["inactivity_days"],
        "engagement_slope":        features["engagement_slope"],
        "assessment_consistency":  features["assessment_consistency"],
    }])
    prob  = float(xgb_model.predict_proba(df)[0][1])
    score = int(prob * 100)
    level = "High" if score >= 70 else ("Medium" if score >= 40 else "Low")
    return {"probability": prob, "risk_score": score, "risk_level": level}


def predict_persona(features, cognitive):
    """Run features through KMeans persona model (same logic as persona_service.py)."""
    focus_score        = float(cognitive.get("focus_score", 50) or 50)
    engagement_score   = float(cognitive.get("engagement_score", 50) or 50)
    confusion_score    = float(cognitive.get("confusion_score", 50) or 50)
    ai_risk_prob       = float(cognitive.get("ai_risk_probability", 0) or 0)

    attention_score    = focus_score
    boredom_score      = max(0, 100 - engagement_score)
    cog_overload_score = ai_risk_prob * 100 if ai_risk_prob <= 1 else ai_risk_prob

    df = pd.DataFrame([{
        "total_clicks":             float(features["total_clicks"]),
        "avg_score":                float(features["avg_score"]),
        "active_days":              float(features["active_days"]),
        "engagement_variability":   float(features["engagement_variability"]),
        "inactivity_days":          float(features["inactivity_days"]),
        "engagement_slope":         float(features["engagement_slope"]),
        "assessment_consistency":   float(features["assessment_consistency"]),
        "attention_score":          attention_score,
        "confusion_score":          confusion_score,
        "boredom_score":            boredom_score,
        "cognitive_overload_score": cog_overload_score,
    }])

    scaled  = persona_scaler.transform(df)
    cluster = int(persona_model.predict(scaled)[0])
    persona = PERSONA_MAP.get(cluster, "Unknown Persona")
    return {"cluster": cluster, "persona": persona}


def risk_heuristic_persona(features, risk_score):
    """Heuristic persona from risk_service.py (lines 231-249). NOT used by pipeline."""
    if risk_score >= 80:
        return "Burnout Pattern"
    elif features["inactivity_days"] >= 20:
        return "Passive Watcher"
    elif features["engagement_slope"] < -0.08:
        return "Anxiety-Spike Learner"
    elif features["avg_score"] < 50:
        return "Struggling Learner"
    else:
        return "Consistent Learner"


def priority_focus(persona, risk_level, cognitive):
    """Mirror of prompt_builder.determine_priority_focus."""
    confusion_score  = float(cognitive.get("confusion_score", 0) or 0)
    engagement_score = float(cognitive.get("engagement_score", 50) or 50)

    if persona == "Consistent Learner" and risk_level in ("Low", "Unknown", None):
        return "enrichment"
    if confusion_score >= 70:
        return "confusion"
    if persona == "Burnout Pattern" and risk_level == "High":
        return "burnout"
    if engagement_score < 50:
        return "engagement"
    return "persona_default"


FRAMEWORK_NAMES = {
    "Burnout Pattern":       "CBT — workload reduction and recovery",
    "Passive Watcher":       "SDT — autonomy and interactive engagement",
    "Anxiety-Spike Learner": "Mindfulness — reassurance and pacing",
    "Silent Isolator":       "Social Presence Theory — connection",
    "Last-Minute Survivor":  "Implementation Intentions — planning",
    "Consistent Learner":    "SDT — enrichment and growth",
}

RISK_DISPLAY = {
    "High":    "Needs Support",
    "Medium":  "On Track",
    "Low":     "Doing Well",
    "Unknown": "Unknown",
}

results_table = []

print("\n" + "=" * 70)
print("  FULL END-TO-END PIPELINE TRACE — ALL THREE EVALUATION SCENARIOS")
print("=" * 70)

for scenario_name, scenario in EVAL_SCENARIOS.items():
    pk                = scenario["profile_key"]
    expected_persona  = scenario["expected_persona"]
    expected_risk     = scenario["expected_risk"]

    profile  = DEMO_PROFILES.get(pk)
    features = profile["features"]
    cognitive = profile["cognitive"]
    seed_label = profile["label"]

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    # ── STAGE 1: Seed Profile ─────────────────────────────────────────────
    print(f"\n[STAGE 1] Seed Profile (seed_service.py → DEMO_PROFILES['{pk}'])")
    print(f"  Seed label  : '{seed_label}'")
    print(f"  Features    : {features}")
    print(f"  Cognitive   : {cognitive}")
    seed_ok = (seed_label == expected_persona)
    print(f"  Label Match : {'PASS' if seed_ok else 'FAIL'} (expected '{expected_persona}')")

    # ── STAGE 2: DB Values ────────────────────────────────────────────────
    print(f"\n[STAGE 2] Database Values")
    print(f"  student_features → {features}")
    print(f"  cognitive_metrics → {cognitive}")

    # ── STAGE 3: Risk Service ─────────────────────────────────────────────
    risk = predict_risk(features)
    risk_persona_heuristic = risk_heuristic_persona(features, risk["risk_score"])
    print(f"\n[STAGE 3] Risk Service (XGBoost)")
    print(f"  probability          : {risk['probability']:.4f}")
    print(f"  risk_score           : {risk['risk_score']}")
    print(f"  risk_level           : '{risk['risk_level']}'")
    print(f"  risk heuristic persona (NOT used by pipeline): '{risk_persona_heuristic}'")

    risk_level = risk["risk_level"]
    risk_ok = True
    if "Low" in expected_risk and "Medium" not in expected_risk and "High" not in expected_risk:
        risk_ok = (risk_level == "Low")
    elif "High" in expected_risk and "Medium" in expected_risk:
        risk_ok = (risk_level in ("High", "Medium"))
    elif "High" in expected_risk:
        risk_ok = (risk_level == "High")

    print(f"  Risk Check           : {'PASS' if risk_ok else 'FAIL'} (expected '{expected_risk}', got '{risk_level}')")

    # ── STAGE 4: Persona Service (KMeans) ────────────────────────────────
    persona_result = predict_persona(features, cognitive)
    ml_persona = persona_result["persona"]
    ml_cluster = persona_result["cluster"]
    print(f"\n[STAGE 4] Persona Service (KMeans + Scaler)")
    print(f"  KMeans Cluster       : {ml_cluster}")
    print(f"  ML Persona           : '{ml_persona}'")
    persona_ok = (ml_persona == expected_persona)
    print(f"  Persona Match        : {'PASS' if persona_ok else 'FAIL'} (expected '{expected_persona}')")
    if not persona_ok:
        print(f"")
        print(f"  !! ROOT CAUSE FOUND:")
        print(f"     FILE   : d:\\EDU AI\\edu-ai\\backend\\services\\seed_service.py")
        print(f"     SECTION: DEMO_PROFILES['{pk}']")
        print(f"     PROBLEM: The feature values seeded for '{pk}' profile map to")
        print(f"              KMeans cluster {ml_cluster} ('{ml_persona}'), NOT cluster for '{expected_persona}'")
        print(f"     FIX    : The feature/cognitive values in seed_service.py need to be")
        print(f"              adjusted so persona_kmeans.pkl correctly assigns '{expected_persona}'")

    # ── STAGE 5: Context Builder ──────────────────────────────────────────
    ctx_persona = ml_persona
    print(f"\n[STAGE 5] Context Builder (context_builder.py)")
    print(f"  context.persona      : '{ctx_persona}'  ← from persona_service.get_student_persona()")
    print(f"  context.risk_level   : '{risk_level}'   ← from risk_service.get_student_risk_service()")

    # ── STAGE 6: Reasoning Layer ──────────────────────────────────────────
    focus_id = priority_focus(ctx_persona, risk_level, cognitive)
    print(f"\n[STAGE 6] Reasoning Layer (reasoning_layer.py → determine_priority_focus)")
    print(f"  Focus ID             : '{focus_id}'")

    # ── STAGE 7: Prompt Builder ───────────────────────────────────────────
    framework = FRAMEWORK_NAMES.get(ctx_persona, "General supportive mentoring")
    print(f"\n[STAGE 7] Prompt Builder (prompt_builder.py)")
    print(f"  Prompt Persona       : '{ctx_persona}'")
    print(f"  Psychological Frame  : {framework}")

    # ── STAGE 8: LLM + API Response ──────────────────────────────────────
    print(f"\n[STAGE 8] LLM Prompt Persona / API Response")
    print(f"  API response.persona : '{ctx_persona}'")
    print(f"  API response.risk_level: '{risk_level}'")

    # ── STAGE 9: UI Display ───────────────────────────────────────────────
    ui_progress = RISK_DISPLAY.get(risk_level, risk_level)
    print(f"\n[STAGE 9] UI Display")
    print(f"  Displayed Persona    : '{ctx_persona}'")
    print(f"  Progress Status      : '{ui_progress}'")

    # ── Overall ───────────────────────────────────────────────────────────
    all_ok = seed_ok and risk_ok and persona_ok
    print(f"\n[RESULT] {'PASS' if all_ok else 'FAIL'}")

    results_table.append({
        "scenario":        scenario_name,
        "expected_persona": expected_persona,
        "ml_persona":      ml_persona,
        "risk_level":      risk_level,
        "risk_score":      risk["risk_score"],
        "ui_persona":      ctx_persona,
        "ui_progress":     ui_progress,
        "focus":           focus_id,
        "persona_ok":      persona_ok,
        "risk_ok":         risk_ok,
        "seed_ok":         seed_ok,
        "status":          "PASS" if all_ok else "FAIL",
        "cluster":         ml_cluster,
        "theme":           framework,
    })


# ─── FINAL TABLE ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FINAL SUMMARY TABLE")
print("=" * 70)
print(f"\n| {'Scenario':<20} | {'Expected Persona':<22} | {'Backend Persona':<22} | {'UI Persona':<22} | {'Response Theme':<35} | Status |")
print("|" + "-"*22 + "|" + "-"*24 + "|" + "-"*24 + "|" + "-"*24 + "|" + "-"*37 + "|--------|")
for r in results_table:
    theme_short = r["theme"][:33]
    print(f"| {r['scenario']:<20} | {r['expected_persona']:<22} | {r['ml_persona']:<22} | {r['ui_persona']:<22} | {theme_short:<35} | {r['status']:^6} |")

print()
any_fail = any(r["status"] == "FAIL" for r in results_table)
if not any_fail:
    print("ALL THREE SCENARIOS PASS — EVALUATION URLs READY FOR SCREENSHOT")
    print("  http://localhost:8080/mentor?eval_mode=true&scenario=low")
    print("  http://localhost:8080/mentor?eval_mode=true&scenario=moderate")
    print("  http://localhost:8080/mentor?eval_mode=true&scenario=high")
else:
    print("SOME SCENARIOS FAIL — ROOT CAUSES IDENTIFIED:")
    print()
    for r in results_table:
        if r["status"] == "FAIL":
            sc = r["scenario"]
            print(f"  SCENARIO {sc}:")
            if not r["persona_ok"]:
                print(f"    [PERSONA MISMATCH]")
                print(f"      Expected  : '{r['expected_persona']}'")
                print(f"      Got       : '{r['ml_persona']}' (cluster {r['cluster']})")
                print(f"      File      : backend/services/seed_service.py")
                print(f"      Function  : DEMO_PROFILES dict (compile-time feature values)")
                print(f"      Why       : The seeded feature/cognitive values do NOT produce")
                print(f"                  the expected KMeans cluster for '{r['expected_persona']}'")
                print(f"      Fix       : Adjust feature values in DEMO_PROFILES so that")
                print(f"                  persona_scaler + persona_kmeans correctly return '{r['expected_persona']}'")
            if not r["risk_ok"]:
                print(f"    [RISK MISMATCH]")
                print(f"      Expected  : '{r['expected_persona']} risk'")
                print(f"      Got       : risk_level='{r['risk_level']}' (score={r['risk_score']})")
                print(f"      File      : backend/services/seed_service.py")
                print(f"      Fix       : Adjust avg_score/inactivity/engagement_slope features")
print("=" * 70)
