from datetime import datetime, timedelta
import random

# =========================
# DEMO DASHBOARD DATA
# =========================
# Returns realistic cohort-level data aligned with the OULAD dataset
# for the psychology presentation, ensuring the dashboard is always
# fully populated.

def get_overview_metrics():
    return {
        "total_students": 32593,
        "high_risk_students": 4128,
        "average_engagement": 68,
        "average_attention": 72
    }


def get_engagement_trend():
    trend = []
    base_date = datetime.utcnow() - timedelta(days=14)
    
    # Generate a realistic 14-day trend line
    eng_base = 75
    att_base = 78
    
    for i in range(14):
        date_str = (base_date + timedelta(days=i)).strftime("%b %d")
        
        # Add some noise
        eng = eng_base + random.randint(-5, 5)
        att = att_base + random.randint(-4, 6)
        
        # Slight downward trend typical in online courses
        eng_base -= 0.5
        att_base -= 0.3
        
        trend.append({
            "date": date_str,
            "engagement": round(eng),
            "attention": round(att)
        })
        
    return trend


def get_persona_distribution():
    return [
        {"name": "Consistent Learner", "value": 45},
        {"name": "Anxiety-Spike Learner", "value": 20},
        {"name": "Passive Watcher", "value": 15},
        {"name": "Burnout Pattern", "value": 10},
        {"name": "Last-Minute Survivor", "value": 7},
        {"name": "Silent Isolator", "value": 3}
    ]
