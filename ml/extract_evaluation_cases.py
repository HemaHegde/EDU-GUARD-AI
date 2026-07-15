"""
EduGuard AI — Human Evaluation Case Extractor
==============================================
Runs the COMPLETE pipeline on the REAL held-out test partition to extract:
  1. Low-Risk student  (argmin predicted probability)
  2. Moderate-Risk student (closest to 0.50 decision boundary)
  3. High-Risk student (argmax predicted probability)

Uses:
  - Frozen XGBoost model  : academic_risk_xgboost.pkl
  - Frozen KMeans model   : persona_kmeans.pkl
  - Frozen Scaler         : persona_scaler.pkl
  - Dataset               : final_student_psychology_dataset.csv

Computes:
  - Predicted probability + risk score + risk level (from risk_service thresholds)
  - Full SHAP values (TreeExplainer) per selected student
  - Persona assignment (KMeans cluster → archetype label)
  - Risk reasons (from risk_service.py conditional logic)
  - Confidence level (from confidence.py rubric)
  - Intervention / Recommendation (from prompt_builder.py _PERSONA_ACTIONS)
  - Psychological framework (from prompt_builder.py _PSYCHOLOGICAL_FRAMEWORKS)
  - Priority focus (from prompt_builder.py determine_priority_focus logic)

Output:
  ieee_results/human_eval_cases.json   — full structured data
  (SHAP waterfall figures already exist: fig_f4, fig_f5, fig_f6)
"""

import os, sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
MODEL_PATH  = os.path.join(SCRIPT_DIR, "academic_risk_xgboost.pkl")
KMEANS_PATH = os.path.join(SCRIPT_DIR, "persona_kmeans.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "persona_scaler.pkl")
OUT_DIR     = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
EVAL_DIR    = os.path.join(SCRIPT_DIR, "human_eval_package")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Feature configuration (matches run_ieee_evaluation.py exactly) ─────────────
FEATURES = [
    "total_clicks",
    "avg_score",
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency",
]

FEATURE_LABELS = {
    "total_clicks":             "Total Interaction Clicks",
    "avg_score":                "Mean Assessment Score",
    "active_days":              "Active Learning Days",
    "engagement_variability":   "Engagement Variability",
    "inactivity_days":          "Inactivity Duration (days)",
    "engagement_slope":         "Engagement Trend (Slope)",
    "assessment_consistency":   "Assessment Consistency (SD)",
}

# Persona map (from advanced_persona_engine.py)
PERSONA_MAP = {
    0: "Silent Isolator",
    1: "Burnout Pattern",
    2: "Anxiety-Spike Learner",
    3: "Passive Watcher",
    4: "Consistent Learner",
    5: "Last-Minute Survivor",
}

# Persona actions (verbatim from prompt_builder.py _PERSONA_ACTIONS)
PERSONA_ACTIONS = {
    "Burnout Pattern": {
        "student": "reduce your workload this week and schedule one real recovery break before your next study session",
        "educator": "flag for a workload/wellness check-in; consider a short extension or reduced load",
    },
    "Passive Watcher": {
        "student": "pick one interactive action today (a short quiz or a discussion post) instead of passive re-reading",
        "educator": "prompt with an autonomy-supportive nudge — offer a choice of task rather than a mandate",
    },
    "Anxiety-Spike Learner": {
        "student": "shorten your next study session and take a short break partway through to reduce pressure",
        "educator": "check in with a reassurance-first message; avoid framing this as a performance warning",
    },
    "Silent Isolator": {
        "student": "ask one question in the course forum or to a classmate this week, even a small one",
        "educator": "reach out proactively — this student is unlikely to initiate contact themselves",
    },
    "Last-Minute Survivor": {
        "student": "set a specific If-Then plan (e.g. 'after dinner, review one topic') to space out revision before the deadline",
        "educator": "consider a scaffolded reminder or interim checkpoint ahead of the next deadline",
    },
    "Consistent Learner": {
        "student": "take on an enrichment activity or advanced topic — current performance supports it",
        "educator": "consider this student for a peer-mentoring or leadership opportunity",
    },
}

DEFAULT_PERSONA_ACTIONS = {
    "student": "review recent course material and reach out with any specific questions",
    "educator": "no strong behavioural pattern is on record yet; monitor and re-check after more activity data is available",
}

# Psychological frameworks (verbatim from prompt_builder.py _PSYCHOLOGICAL_FRAMEWORKS)
PSYCHOLOGICAL_FRAMEWORKS = {
    "Burnout Pattern": (
        "CBT — workload reduction and recovery: break tasks into tiny steps, "
        "challenge all-or-nothing thinking, explicitly recommend reducing load this week "
        "and building in a recovery break; flag for educator outreach rather than pushing more output."
    ),
    "Passive Watcher": (
        "SDT — autonomy and interactive engagement: connect material to real-world relevance, "
        "offer a choice of next step, use Socratic questions, and nudge toward one small interactive "
        "action (a quiz, a discussion post) instead of passive review."
    ),
    "Anxiety-Spike Learner": (
        "Mindfulness — reassurance and pacing: validate the overwhelm first, use a grounding technique, "
        "recommend shorter study sessions with breaks, and reinforce confidence by naming a concrete "
        "recent strength before assigning anything new."
    ),
    "Silent Isolator": (
        "Social Presence Theory — connection: build rapport, actively invite them to ask a question "
        "or share where they're stuck, suggest one low-stakes social/forum interaction, "
        "and flag for proactive educator outreach given the isolation pattern."
    ),
    "Last-Minute Survivor": (
        "Implementation Intentions — planning: propose a concrete If-Then study micro-habit "
        "tied to a specific deadline, recommend spaced revision over cramming, "
        "and name the cognitive-load cost of leaving it to the last minute."
    ),
    "Consistent Learner": (
        "SDT — enrichment and growth: affirm what is already working, then offer a stretch opportunity "
        "— advanced material, a peer-mentoring or leadership role — rather than remedial advice; "
        "this student does not need risk-reduction framing."
    ),
}

# ── STEP 1: Load data and model ────────────────────────────────────────────────
print("=" * 65)
print("EduGuard AI — Human Evaluation Case Extractor")
print("=" * 65)
print(f"\n[1] Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"    Dataset shape: {df.shape}")

X = df[FEATURES].copy()
y = df["psychological_risk"].copy()

# Reproduce EXACT same split as the IEEE evaluation (random_state=42, stratify=y, test_size=0.20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Also keep the original DataFrame indices to recover student IDs
idx_train, idx_test = train_test_split(
    df.index, test_size=0.20, random_state=42, stratify=y
)
df_test = df.loc[idx_test].reset_index(drop=False)  # keep original index

print(f"    Test partition size: {len(X_test)} students")

print(f"\n[2] Loading frozen XGBoost model: {MODEL_PATH}")
xgb_model = joblib.load(MODEL_PATH)
print("    Model loaded OK.")

# ── STEP 2: Run predictions on test partition ──────────────────────────────────
print("\n[3] Running XGBoost predictions on test partition...")
probs = xgb_model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.50).astype(int)
print(f"    Predictions complete. Min={probs.min():.4f}, Max={probs.max():.4f}")

# ── STEP 3: Select three representative students ───────────────────────────────
print("\n[4] Selecting three representative students...")

low_risk_idx  = int(np.argmin(probs))           # lowest probability → clearest Low Risk
high_risk_idx = int(np.argmax(probs))           # highest probability → clearest High Risk
mod_risk_idx  = int(np.argsort(np.abs(probs - 0.50))[0])  # closest to 0.50 → Moderate Risk

cases = {
    "Low Risk":      low_risk_idx,
    "Moderate Risk": mod_risk_idx,
    "High Risk":     high_risk_idx,
}

for case_name, cidx in cases.items():
    print(f"    {case_name:15s} -> test_idx={cidx:5d}, prob={probs[cidx]:.6f}, "
          f"pred_class={preds[cidx]}, actual={y_test.iloc[cidx]}")

# ── STEP 4: Compute SHAP values for the FULL test set (needed for waterfall) ──
print("\n[5] Computing SHAP values (TreeExplainer) on test set...")
explainer   = shap.TreeExplainer(xgb_model)
X_test_labeled = X_test.copy()
X_test_labeled.columns = [FEATURE_LABELS[c] for c in X_test_labeled.columns]
shap_values = explainer(X_test_labeled)
print("    SHAP values computed.")

# ── STEP 5: Load KMeans persona model ─────────────────────────────────────────
print(f"\n[6] Loading persona profiles from pre-computed CSV (advanced_persona_profiles.csv)...")
PERSONA_PROFILES_PATH = os.path.join(SCRIPT_DIR, "advanced_persona_profiles.csv")
persona_profiles_df = pd.read_csv(PERSONA_PROFILES_PATH)
print(f"    Loaded {len(persona_profiles_df)} pre-computed persona profiles.")
# Persona assignments already pre-computed. Match test students by feature values.

# ── STEP 6: Risk reasons logic (from risk_service.py, verbatim) ───────────────
def get_risk_reasons(row):
    reasons = []
    if row["inactivity_days"] > 15:
        reasons.append(f"High inactivity detected ({int(row['inactivity_days'])} days)")
    if row["engagement_slope"] < -0.05:
        reasons.append("Student engagement trend declining")
    if row["total_clicks"] < 100:
        reasons.append("Low learning interaction activity")
    if row["avg_score"] < 50:
        reasons.append("Academic performance weakening")
    if row["assessment_consistency"] > 12:
        reasons.append("Irregular assessment consistency detected")
    if row["engagement_variability"] > 8:
        reasons.append("Behavior fluctuation increasing")
    if len(reasons) == 0:
        reasons.append("Student learning behavior stable")
    return reasons

# ── STEP 7: Risk level thresholds (from risk_service.py, verbatim) ────────────
def get_risk_level(risk_score):
    if risk_score >= 70:
        return "High"
    elif risk_score >= 40:
        return "Moderate"
    else:
        return "Low"

# ── STEP 8: Persona detection (from risk_service.py heuristic, verbatim) ──────
def get_persona_heuristic(row, risk_score):
    """Heuristic from risk_service.py (used in live API fallback)"""
    if risk_score >= 80:
        return "Burnout Pattern"
    elif row["inactivity_days"] >= 20:
        return "Passive Watcher"
    elif row["engagement_slope"] < -0.08:
        return "Anxiety-Spike Learner"
    elif row["avg_score"] < 50:
        return "Struggling Learner"
    else:
        return "Consistent Learner"

# ── STEP 9: Priority focus determination (from prompt_builder.py, verbatim) ───
def determine_priority_focus(persona, risk_level, risk_score, row):
    """
    Deterministic priority cascade from prompt_builder.py determine_priority_focus().
    Cognitive state is unavailable here (no live cognitive module in offline script),
    so cognitive branches are skipped — reported accurately in validation notes.
    """
    actions = PERSONA_ACTIONS.get(persona, DEFAULT_PERSONA_ACTIONS)

    # 1. Consistent Learner + non-elevated risk → enrichment guardrail
    if persona == "Consistent Learner" and risk_level in ("Low", "Unknown"):
        return {
            "id": "enrichment",
            "instruction": (
                "This learner is stable and not at elevated risk — lead with enrichment "
                "or a leadership opportunity. Do not use remedial or risk-reduction framing "
                "anywhere in your reply."
            ),
            "reason": f"persona=Consistent Learner, risk_level={risk_level}",
            "student_action": actions["student"],
            "educator_action": actions["educator"],
        }

    # 2. Cognitive confusion — SKIPPED (no live cognitive module in extraction script)

    # 3. Severe burnout: persona AND High risk together
    if persona == "Burnout Pattern" and risk_level == "High":
        return {
            "id": "burnout",
            "instruction": (
                "Burnout risk is severe — prioritize workload reduction and recovery "
                "over anything else; do not assign new work."
            ),
            "reason": "persona=Burnout Pattern, risk_level=High",
            "student_action": actions["student"],
            "educator_action": actions["educator"],
        }

    # 4. Low engagement — SKIPPED (no live cognitive module)

    # 5. SHAP-identified inactivity (check if top SHAP feature is inactivity, direction = +risk)
    # We check by inspecting the row's feature values
    if row["inactivity_days"] > 15:
        return {
            "id": "inactivity",
            "instruction": (
                "Inactivity is the strongest measured risk driver — anchor advice on "
                "resuming regular activity, not general study tips."
            ),
            "reason": f"inactivity_days={row['inactivity_days']:.1f} (>15 days threshold)",
            "student_action": (
                "log in and complete one small piece of coursework today to break the "
                "inactivity streak, then build back up gradually"
            ),
            "educator_action": (
                "reach out directly — inactivity is the strongest measured driver of this student's risk"
            ),
        }

    # 6. Default: persona strategy carries response
    return {
        "id": "persona_default",
        "instruction": (
            f"No single signal dominates this evidence set — let the {persona} strategy "
            f"above guide tone and focus."
        ),
        "reason": "no dominant threshold was crossed",
        "student_action": actions["student"],
        "educator_action": actions["educator"],
    }

# ── STEP 10: Confidence level (from confidence.py rubric, verbatim) ───────────
def assess_confidence_level(has_risk_score, has_persona, has_shap, has_cognitive=False):
    """
    Simplified offline version of confidence.py assess_confidence().
    Returns level + rationale. SHAP is always available in this script.
    Cognitive state is unavailable (no live cognitive module) — reported accurately.
    """
    if has_risk_score and has_persona and has_shap:
        level = "High"
        rationale = (
            "Risk score, SHAP-based feature explanation, and a known learner persona "
            "are all available, giving a quantified, explained, and contextualized "
            "basis for this response."
        )
    elif has_risk_score and has_persona:
        level = "Medium"
        rationale = (
            "Risk score and learner persona are available, but no SHAP explanation "
            "was available to confirm the specific drivers behind the risk score."
        )
    else:
        level = "Low"
        rationale = (
            "Limited evidence available. Response should be treated as general "
            "guidance rather than a personalized, evidence-backed recommendation."
        )
    if has_cognitive:
        rationale += " Cognitive-state data was also available as a supporting signal."
    return level, rationale

# ── STEP 11: Build complete profile for each selected student ──────────────────
print("\n[7] Building complete profiles for selected students...")

CASE_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]
CASE_INDICES = [low_risk_idx, mod_risk_idx, high_risk_idx]
ANONYMIZED_IDS = ["S-LR-001", "S-MR-002", "S-HR-003"]

results = {}

for label, cidx, anon_id in zip(CASE_LABELS, CASE_INDICES, ANONYMIZED_IDS):
    print(f"\n  --- {label} (test index {cidx}) ---")

    # Raw feature values
    row = X_test.iloc[cidx]
    row_dict = {feat: float(row[feat]) for feat in FEATURES}
    prob      = float(probs[cidx])
    pred_cls  = int(preds[cidx])
    actual_y  = int(y_test.iloc[cidx])
    risk_score_val = int(prob * 100)
    risk_level_str = get_risk_level(risk_score_val)

    print(f"    Probability: {prob:.6f} | Risk Score: {risk_score_val} | Level: {risk_level_str}")
    print(f"    Predicted Class: {pred_cls} | Actual Label: {actual_y}")

    # Feature values for display
    feat_display = {FEATURE_LABELS[f]: round(row_dict[f], 4) for f in FEATURES}

    # Risk reasons
    risk_reasons = get_risk_reasons(row_dict)
    print(f"    Risk Reasons: {risk_reasons}")

    # Persona from pre-computed advanced_persona_profiles.csv
    # Match by total_clicks + avg_score + active_days + inactivity_days (exact float match)
    match_mask = (
        (persona_profiles_df["total_clicks"].round(4) == round(row_dict["total_clicks"], 4)) &
        (persona_profiles_df["avg_score"].round(4)    == round(row_dict["avg_score"], 4)) &
        (persona_profiles_df["active_days"].round(4)  == round(row_dict["active_days"], 4)) &
        (persona_profiles_df["inactivity_days"].round(4) == round(row_dict["inactivity_days"], 4))
    )
    matched = persona_profiles_df[match_mask]
    if len(matched) >= 1:
        cluster_id     = int(matched.iloc[0]["persona_cluster"])
        persona_kmeans = str(matched.iloc[0]["persona"])
        cognitive_row  = matched.iloc[0]
        has_cog        = True
        cog_state = {
            "attention_score":          float(cognitive_row["attention_score"]),
            "confusion_score":          float(cognitive_row["confusion_score"]),
            "boredom_score":            float(cognitive_row["boredom_score"]),
            "cognitive_overload_score": float(cognitive_row["cognitive_overload_score"]),
            "stability_score":          float(cognitive_row["stability_score"]),
        }
    else:
        # No exact match — use heuristic fallback
        cluster_id     = -1
        persona_kmeans = get_persona_heuristic(row_dict, risk_score_val)
        has_cog        = False
        cog_state      = {}
        print(f"    WARNING: No exact feature match in persona profiles — using heuristic persona")
    # Also get the heuristic persona (from risk_service.py live API) for comparison
    persona_heuristic = get_persona_heuristic(row_dict, risk_score_val)
    print(f"    KMeans Persona: {persona_kmeans} (cluster {cluster_id})")
    print(f"    Heuristic Persona (risk_service.py): {persona_heuristic}")

    # Use KMeans persona as the primary (it's from the frozen clustering model)
    persona_used = persona_kmeans

    # SHAP values for this student
    sv = shap_values[cidx]
    shap_vals_arr   = sv.values        # shape: (n_features,)
    shap_feat_names = list(X_test_labeled.columns)
    shap_base_value = float(sv.base_values)

    shap_contributions = []
    for i, feat_name in enumerate(shap_feat_names):
        feat_value = float(X_test_labeled.iloc[cidx][feat_name])
        shap_val   = float(shap_vals_arr[i])
        shap_contributions.append({
            "feature": feat_name,
            "feature_value": round(feat_value, 4),
            "shap_value": round(shap_val, 6),
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })

    # Sort by absolute SHAP descending
    shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Top positive (risk-increasing) SHAP
    top_positive = [s for s in shap_contributions if s["shap_value"] > 0]
    # Top negative (risk-decreasing / protective) SHAP
    top_negative = [s for s in shap_contributions if s["shap_value"] < 0]

    print(f"    SHAP base value: {shap_base_value:.4f}")
    print(f"    Top risk-increasing: {[(s['feature'], round(s['shap_value'],3)) for s in top_positive[:3]]}")
    print(f"    Top protective:      {[(s['feature'], round(s['shap_value'],3)) for s in top_negative[:3]]}")

    # Confidence level
    has_risk    = True  # always available (model ran)
    has_persona = persona_used not in (None, "", "Unknown Persona")
    has_shap    = True  # always available (TreeExplainer ran)
    # has_cog is set above in the persona lookup block
    conf_level, conf_rationale = assess_confidence_level(has_risk, has_persona, has_shap, has_cog)
    print(f"    Confidence Level: {conf_level}")

    # Human review status
    # needs_human_review fires if level == "Low" or normalized_score < 0.50
    # In offline mode without retrieval: normalized_score = 0.0 always < 0.50
    # → needs_human_review = True for all cases in offline mode
    # Report this accurately with full explanation
    needs_human_review = True  # because normalized_score=0.0 (no retrieval available offline)
    human_review_reason = (
        "needs_human_review=True: retrieval pipeline not available in offline extraction mode "
        "(normalized_score=0.0 < threshold 0.50). In live API, High-confidence cases may not "
        "require human review if retrieval similarity is sufficient."
    )

    # Priority focus
    focus = determine_priority_focus(persona_used, risk_level_str, risk_score_val, row_dict)

    # Persona actions + framework
    actions   = PERSONA_ACTIONS.get(persona_used, DEFAULT_PERSONA_ACTIONS)
    framework = PSYCHOLOGICAL_FRAMEWORKS.get(persona_used, "General supportive mentoring.")

    # Behavioral summary
    behavioral_summary = []
    if row_dict["active_days"] > 20:
        behavioral_summary.append(f"Actively logged in on {int(row_dict['active_days'])} days (high engagement)")
    elif row_dict["active_days"] > 10:
        behavioral_summary.append(f"Moderately active: {int(row_dict['active_days'])} login days")
    else:
        behavioral_summary.append(f"Very low activity: only {int(row_dict['active_days'])} active days")

    if row_dict["inactivity_days"] > 15:
        behavioral_summary.append(f"Extended inactivity: {int(row_dict['inactivity_days'])} days absent")
    elif row_dict["inactivity_days"] > 5:
        behavioral_summary.append(f"Moderate inactivity: {int(row_dict['inactivity_days'])} days absent")
    else:
        behavioral_summary.append(f"Minimal inactivity: {int(row_dict['inactivity_days'])} days absent")

    if row_dict["avg_score"] >= 70:
        behavioral_summary.append(f"Strong academic performance (avg score: {row_dict['avg_score']:.1f})")
    elif row_dict["avg_score"] >= 50:
        behavioral_summary.append(f"Adequate academic performance (avg score: {row_dict['avg_score']:.1f})")
    else:
        behavioral_summary.append(f"Weakening academic performance (avg score: {row_dict['avg_score']:.1f})")

    if row_dict["engagement_slope"] > 0.02:
        behavioral_summary.append("Engagement trend: improving over time")
    elif row_dict["engagement_slope"] < -0.05:
        behavioral_summary.append("Engagement trend: declining over time")
    else:
        behavioral_summary.append("Engagement trend: roughly stable")

    if row_dict["assessment_consistency"] < 8:
        behavioral_summary.append(f"Assessment performance consistent (SD={row_dict['assessment_consistency']:.1f})")
    elif row_dict["assessment_consistency"] < 15:
        behavioral_summary.append(f"Assessment performance moderately inconsistent (SD={row_dict['assessment_consistency']:.1f})")
    else:
        behavioral_summary.append(f"Assessment performance highly inconsistent (SD={row_dict['assessment_consistency']:.1f})")

    total_clicks = int(row_dict["total_clicks"])
    if total_clicks >= 500:
        behavioral_summary.append(f"High VLE interaction ({total_clicks:,} total clicks)")
    elif total_clicks >= 100:
        behavioral_summary.append(f"Moderate VLE interaction ({total_clicks:,} total clicks)")
    else:
        behavioral_summary.append(f"Low VLE interaction ({total_clicks:,} total clicks)")

    # Assemble full result record
    results[label] = {
        "anonymized_id": anon_id,
        "test_partition_index": cidx,
        "risk": {
            "predicted_probability": round(prob, 6),
            "risk_score_0_100": risk_score_val,
            "risk_level": risk_level_str,
            "predicted_class": pred_cls,
            "actual_label": actual_y,
            "correct_prediction": pred_cls == actual_y,
            "risk_reasons": risk_reasons,
        },
        "features": {
            "raw": row_dict,
            "display": feat_display,
        },
        "persona": {
            "kmeans_cluster_id": cluster_id,
            "kmeans_persona_label": persona_kmeans,
            "heuristic_persona_label": persona_heuristic,
            "persona_used": persona_used,
            "intervention_style": {
                "Silent Isolator": "Social engagement support",
                "Burnout Pattern": "Mental wellness intervention",
                "Anxiety-Spike Learner": "Guided pacing and stress reduction",
                "Passive Watcher": "Interactive participation encouragement",
                "Consistent Learner": "Advanced learning opportunities",
                "Last-Minute Survivor": "Deadline management coaching",
            }.get(persona_used, "Monitoring and general support"),
        },
        "shap": {
            "base_value": round(shap_base_value, 6),
            "predicted_log_odds_contribution": round(sum(shap_vals_arr), 6),
            "all_contributions": shap_contributions,
            "top_3_risk_increasing": top_positive[:3],
            "top_3_protective": top_negative[:3],
        },
        "cognitive_state": {
            "available": has_cog,
            "data": cog_state if has_cog else None,
            "source": (
                "Pre-computed from advanced_persona_profiles.csv (synthetic mapping of cognitive "
                "scores from the EdNet video intelligence pipeline to OULAD students via repeated "
                "tile mapping, as documented in advanced_persona_engine.py)."
                if has_cog else
                "No match found in pre-computed profiles — cognitive state unavailable for this student."
            ),
            "interpretation": {
                "attention_score":          "Higher = more attentive during learning sessions",
                "confusion_score":          "Higher = more confused (>70 triggers confusion priority focus)",
                "boredom_score":            "Higher = more bored/disengaged during sessions",
                "cognitive_overload_score": "Higher = more cognitively overloaded",
                "stability_score":          "Higher = more behaviourally stable (0-100, derived composite)",
            } if has_cog else None,
        },
        "confidence": {
            "level": conf_level,
            "rationale": conf_rationale,
            "has_risk_score": has_risk,
            "has_persona": has_persona,
            "has_shap": has_shap,
            "has_cognitive_state": has_cog,
            "needs_human_review": needs_human_review,
            "needs_human_review_reason": human_review_reason,
        },
        "priority_focus": focus,
        "recommendations": {
            "student_recommendation": actions["student"],
            "educator_recommendation": actions["educator"],
            "psychological_framework": framework,
            "conversation_strategy": {
                "enrichment": {
                    "opening_goal": "Affirm what's already working",
                    "teaching_style": "Forward-looking and growth-oriented; no risk or remedial framing",
                    "recommendation_style": "One stretch/enrichment opportunity",
                    "educator_goal": "Consider for a peer-mentoring or leadership role",
                },
                "burnout": {
                    "opening_goal": "Reassure and validate; do not push more work",
                    "teaching_style": "Brief, low-pressure, permission-giving tone",
                    "recommendation_style": "One recovery-oriented micro-action",
                    "educator_goal": "Flag for a workload/wellness check-in",
                },
                "inactivity": {
                    "opening_goal": "Welcome back, non-judgmental",
                    "teaching_style": "Focus on restarting momentum, not catching up all at once",
                    "recommendation_style": "One tiny piece of coursework today, to break the streak",
                    "educator_goal": "Reach out directly given the inactivity pattern",
                },
                "persona_default": {
                    "opening_goal": "Lead with the learner's own behavioural pattern",
                    "teaching_style": framework,
                    "recommendation_style": "One action grounded in the persona strategy above",
                    "educator_goal": "Monitor and re-check after more activity data is available",
                },
            }.get(focus["id"], {
                "opening_goal": f"Apply {persona_used} strategy",
                "teaching_style": framework,
                "recommendation_style": "One persona-appropriate action",
                "educator_goal": "Monitor and engage based on persona indicators",
            }),
        },
        "behavioral_summary": behavioral_summary,
        "figures": {
            "shap_waterfall": {
                "Low Risk":      "ieee_results/figures/fig_f5_shap_waterfall_lowrisk.png",
                "Moderate Risk": "ieee_results/figures/fig_f6_shap_waterfall_borderline.png",
                "High Risk":     "ieee_results/figures/fig_f4_shap_waterfall_highrisk.png",
            }[label],
            "shap_importance_global": "ieee_results/figures/fig3_shap_importance.png",
            "shap_beeswarm": "ieee_results/figures/fig4_shap_beeswarm.png",
            "persona_clusters_tsne": "ieee_results/figures/fig_g2_tsne_clusters.png",
        },
    }

# ── STEP 12: Generate per-student SHAP waterfall figures ──────────────────────
print("\n[8] Generating SHAP waterfall figures for each selected student...")

WATERFALL_PATHS = {
    "Low Risk":      os.path.join(FIG_DIR, "eval_waterfall_low_risk.png"),
    "Moderate Risk": os.path.join(FIG_DIR, "eval_waterfall_moderate_risk.png"),
    "High Risk":     os.path.join(FIG_DIR, "eval_waterfall_high_risk.png"),
}

for label, cidx in zip(CASE_LABELS, CASE_INDICES):
    prob_val = probs[cidx]
    risk_score_val = int(prob_val * 100)
    risk_level_str = get_risk_level(risk_score_val)
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[cidx], max_display=7, show=False)
    plt.title(
        f"SHAP Waterfall — {label} Student | Anonymized ID: {ANONYMIZED_IDS[CASE_LABELS.index(label)]}\n"
        f"Predicted Probability: {prob_val:.4f} | Risk Score: {risk_score_val}/100 | Level: {risk_level_str}",
        fontsize=10, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    plt.savefig(WATERFALL_PATHS[label], dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [SAVED] {WATERFALL_PATHS[label]}")
    # Update figure path in results
    results[label]["figures"]["shap_waterfall_generated"] = WATERFALL_PATHS[label]

# ── STEP 13: Save results JSON ─────────────────────────────────────────────────
print("\n[9] Saving results to JSON...")

class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

json_path = os.path.join(OUT_DIR, "human_eval_cases.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print(f"    [SAVED] {json_path}")

# ── STEP 14: Print Summary Table ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EXTRACTION COMPLETE — SUMMARY TABLE")
print("=" * 65)
print(f"\n  {'Case':<15} {'ID':<12} {'Prob':>8} {'Score':>6} {'Level':>10} {'Persona':<25} {'Conf':>7}")
print("  " + "-" * 80)
for label, anon_id in zip(CASE_LABELS, ANONYMIZED_IDS):
    r = results[label]
    print(
        f"  {label:<15} {anon_id:<12} "
        f"{r['risk']['predicted_probability']:>8.4f} "
        f"{r['risk']['risk_score_0_100']:>6d} "
        f"{r['risk']['risk_level']:>10} "
        f"{r['persona']['persona_used']:<25} "
        f"{r['confidence']['level']:>7}"
    )

print("\n[10] Done. All outputs saved to:")
print(f"     JSON  : {json_path}")
for label, path in WATERFALL_PATHS.items():
    print(f"     Figure: {path}")
print()
