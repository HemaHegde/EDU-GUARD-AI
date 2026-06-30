import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
COG_PATH = os.path.join(SCRIPT_DIR, "video_intelligence", "cognitive_behavior_results.csv")

# 1. Load Data
df = pd.read_csv(DATA_PATH)
cog_df = pd.read_csv(COG_PATH)

# Align cognitive data to main dataset (repeat if necessary)
repeat = int(len(df) / len(cog_df)) + 1
cog_expanded = pd.concat([cog_df] * repeat, ignore_index=True)
cog_expanded = cog_expanded.iloc[:len(df)].reset_index(drop=True)

# Add cognitive overload score to main df
df["gru_cognitive_overload"] = cog_expanded["cognitive_overload_score"]

# 7 behavioural features
FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

# Top-4 SHAP features
SHAP_FEATURES = [
    "inactivity_days", "active_days", "avg_score", "assessment_consistency"
]

X_behav = df[FEATURES].values

# 2. Compute K-Means archetype label (k=6)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_behav)
km = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=300)
df["archetype_label"] = km.fit_predict(X_sc)

y = df["psychological_risk"].values

# Split data (using stratify=y and random_state=42)
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
df_train = df.loc[train_idx]
df_test = df.loc[test_idx]
y_train = df_train["psychological_risk"].values
y_test = df_test["psychological_risk"].values

# 3. Define the configs
configs = {
    "Config A (Baseline - 7 features)": FEATURES,
    "Config B (SHAP Top-4)": SHAP_FEATURES,
    "Config C (7 features + archetype label)": FEATURES + ["archetype_label"],
    "Config D (7 features + GRU score)": FEATURES + ["gru_cognitive_overload"],
    "Config E (9 features total)": FEATURES + ["archetype_label", "gru_cognitive_overload"],
}

# XGBoost default setup
def train_eval(features):
    X_train = df_train[features].fillna(0).values
    X_test = df_test[features].fillna(0).values
    
    # Train
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Eval
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    
    return acc, pre, rec, f1, auc

# Run all
results = []
baseline_auc = None

for name, feats in configs.items():
    acc, pre, rec, f1, auc = train_eval(feats)
    if name.startswith("Config A"):
        baseline_auc = auc
        delta_auc = 0.0
    else:
        delta_auc = auc - baseline_auc
        
    results.append({
        "Config": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc,
        "Delta AUC vs Config A": delta_auc
    })

# Format and print clean table
print(f"{'Configuration':<45} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6} | {'Delta AUC':<8}")
print("-" * 100)
for r in results:
    print(f"{r['Config']:<45} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC-AUC']:.4f} | {r['Delta AUC vs Config A']:+.4f}")
