import pandas as pd
import os
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../ml/academic_risk_xgboost.pkl")

# Load model globally to avoid loading it on every request
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        # XGBoost models usually work well with TreeExplainer
        explainer = shap.TreeExplainer(model)
        print("SHAP Explainer Loaded Successfully")
    else:
        model = None
        explainer = None
        print("Risk Model for SHAP Not Found")
except Exception as e:
    model = None
    explainer = None
    print(f"Error loading SHAP Explainer: {e}")

def explain_student_psychology(features_dict: dict):
    """
    Generates SHAP values for the given student features to scientifically explain
    which behavioral factors are contributing to their psychological/risk state.
    """
    if explainer is None or model is None:
        return {"status": "error", "message": "SHAP explainer not initialized"}

    try:
        # Create a DataFrame from the feature dictionary (must match training feature order)
        # Expected features: total_clicks, avg_score, active_days, engagement_variability, inactivity_days, engagement_slope, assessment_consistency
        features_df = pd.DataFrame([features_dict])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        # SHAP values for the single student (we take the first row)
        # For binary classification with XGBoost, shap_values might be a list or array.
        # Usually, shap_values is a 2D array (samples x features)
        student_shap = shap_values[0]

        # Map features to their SHAP impacts
        feature_names = features_df.columns.tolist()
        impacts = []
        
        for idx, feature_name in enumerate(feature_names):
            shap_val = float(student_shap[idx])
            impacts.append({
                "feature": feature_name,
                "value": float(features_df.iloc[0, idx]),
                "shap_value": shap_val,
                "impact_direction": "positive" if shap_val > 0 else "negative",
                "impact_magnitude": abs(shap_val)
            })

        # Sort by highest magnitude (most influential features)
        impacts.sort(key=lambda x: x["impact_magnitude"], reverse=True)

        return {
            "status": "success",
            "base_value": float(explainer.expected_value),
            "shap_explanations": impacts
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_student_explanation(user_id: str):
    from config.supabase_client import supabase
    
    try:
        feature_response = supabase.table("student_features").select("*").eq("user_id", user_id).execute()
        if not feature_response.data:
            return {"status": "error", "message": "No engineered features found for student"}

        features_db = feature_response.data[0]
        
        features_dict = {
            "total_clicks": features_db.get("total_clicks", 0) or 0,
            "avg_score": features_db.get("avg_score", 0) or 0,
            "active_days": features_db.get("active_days", 0) or 0,
            "engagement_variability": features_db.get("engagement_variability", 0) or 0,
            "inactivity_days": features_db.get("inactivity_days", 0) or 0,
            "engagement_slope": features_db.get("engagement_slope", 0) or 0,
            "assessment_consistency": features_db.get("assessment_consistency", 0) or 0
        }
        
        return explain_student_psychology(features_dict)
    except Exception as e:
        return {"status": "error", "message": str(e)}
