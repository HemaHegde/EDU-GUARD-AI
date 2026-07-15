import sys, os, joblib, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
KMEANS_PATH  = os.path.join(BASE, "../ml/persona_kmeans.pkl")
SCALER_PATH  = os.path.join(BASE, "../ml/persona_scaler.pkl")
XGBOOST_PATH = os.path.join(BASE, "../ml/academic_risk_xgboost.pkl")

persona_model  = joblib.load(KMEANS_PATH)
persona_scaler = joblib.load(SCALER_PATH)
xgb_model      = joblib.load(XGBOOST_PATH)

base_features = {
    'total_clicks': 2275, 
    'avg_score': 85.6, 
    'active_days': 78, 
    'engagement_variability': 20.51, 
    'inactivity_days': 160, 
    'engagement_slope': -0.21, 
    'assessment_consistency': 10.9
}

def try_features(feats):
    df = pd.DataFrame([feats])
    prob = float(xgb_model.predict_proba(df)[0][1])
    score = int(prob * 100)
    
    cog = {'focus_score': 85, 'engagement_score': 100, 'confusion_score': 14, 'ai_risk_probability': prob}
    df_kmeans = pd.DataFrame([{
        'total_clicks': feats['total_clicks'],
        'avg_score': feats['avg_score'],
        'active_days': feats['active_days'],
        'engagement_variability': feats['engagement_variability'],
        'inactivity_days': feats['inactivity_days'],
        'engagement_slope': feats['engagement_slope'],
        'assessment_consistency': feats['assessment_consistency'],
        'attention_score': cog['focus_score'],
        'confusion_score': cog['confusion_score'],
        'boredom_score': 0,
        'cognitive_overload_score': prob * 100 if prob <= 1 else prob
    }])
    cluster = int(persona_model.predict(persona_scaler.transform(df_kmeans))[0])
    print(f"Features: avg_score={feats['avg_score']}, slope={feats['engagement_slope']}, inactivity={feats['inactivity_days']} => Score: {score} (Prob: {prob:.4f}), Cluster: {cluster}")

f = base_features.copy(); try_features(f)
f = base_features.copy(); f['avg_score'] = 60.0; try_features(f)
f = base_features.copy(); f['inactivity_days'] = 220; try_features(f)
f = base_features.copy(); f['engagement_slope'] = -0.5; try_features(f)
f = base_features.copy(); f['avg_score'] = 70.0; f['engagement_slope'] = -0.3; try_features(f)
f = base_features.copy(); f['avg_score'] = 65.0; f['engagement_slope'] = -0.25; f['inactivity_days'] = 180; try_features(f)
f = base_features.copy(); f['avg_score'] = 40.0; f['engagement_slope'] = -0.4; try_features(f)
