import pandas as pd
import numpy as np

# =========================
# MACHINE LEARNING IMPORTS
# =========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

import shap

# =========================
# LOAD DATASETS
# =========================

print("Loading OULAD datasets...")

student_info = pd.read_csv("../datasets/studentInfo.csv")
student_vle = pd.read_csv("../datasets/studentVle.csv")
student_assessment = pd.read_csv("../datasets/studentAssessment.csv")

# =========================
# BASIC DATA INFORMATION
# =========================

print("\nStudent Info Shape:")
print(student_info.shape)

print("\nStudent VLE Shape:")
print(student_vle.shape)

print("\nStudent Assessment Shape:")
print(student_assessment.shape)

# =========================
# DATA PREVIEW
# =========================

print("\nStudent Info Preview:")
print(student_info.head())

print("\nStudent VLE Preview:")
print(student_vle.head())

print("\nStudent Assessment Preview:")
print(student_assessment.head())

# =========================
# FEATURE ENGINEERING
# =========================

print("\nCreating engagement features...")

engagement_features = (
    student_vle
    .groupby("id_student")["sum_click"]
    .sum()
    .reset_index()
)

engagement_features.rename(
    columns={
        "sum_click": "total_clicks"
    },
    inplace=True
)

print("\nEngagement Features Preview:")
print(engagement_features.head())

# =========================
# ASSESSMENT FEATURES
# =========================

print("\nCreating assessment features...")

assessment_features = (
    student_assessment
    .groupby("id_student")["score"]
    .mean()
    .reset_index()
)

assessment_features.rename(
    columns={
        "score": "avg_score"
    },
    inplace=True
)

print("\nAssessment Features Preview:")
print(assessment_features.head())

# =========================
# PSYCHOLOGICAL RISK LABELS
# =========================

print("\nCreating psychological risk labels...")

student_info["psychological_risk"] = (
    student_info["final_result"]
    .apply(
        lambda x:
        1 if x in ["Withdrawn", "Fail"]
        else 0
    )
)

print("\nPsychological Risk Label Distribution:")
print(
    student_info["psychological_risk"]
    .value_counts()
)

# =========================
# MERGE DATASETS
# =========================

print("\nMerging datasets...")

merged_data = student_info.merge(
    engagement_features,
    on="id_student",
    how="left"
)

merged_data = merged_data.merge(
    assessment_features,
    on="id_student",
    how="left"
)

# =========================
# HANDLE NULL VALUES
# =========================

merged_data["total_clicks"] = (
    merged_data["total_clicks"]
    .fillna(0)
)

merged_data["avg_score"] = (
    merged_data["avg_score"]
    .fillna(0)
)

# =========================
# DATA PREVIEW
# =========================

print("\nMerged Dataset Preview:")

print(
    merged_data[
        [
            "id_student",
            "total_clicks",
            "avg_score",
            "psychological_risk"
        ]
    ].head()
)

print("\nFinal Dataset Shape:")
print(merged_data.shape)

print("\nPreprocessing completed successfully!")

# =========================
# TEMPORAL FEATURES
# =========================

print("\nCreating temporal engagement features...")

# -------------------------
# ACTIVE DAYS
# -------------------------

active_days = (
    student_vle
    .groupby("id_student")["date"]
    .nunique()
    .reset_index()
)

active_days.rename(
    columns={
        "date": "active_days"
    },
    inplace=True
)

print("\nActive Days Preview:")
print(active_days.head())

# -------------------------
# ENGAGEMENT VARIABILITY
# -------------------------

engagement_variability = (
    student_vle
    .groupby("id_student")["sum_click"]
    .std()
    .reset_index()
)

engagement_variability.rename(
    columns={
        "sum_click":
        "engagement_variability"
    },
    inplace=True
)

engagement_variability[
    "engagement_variability"
] = (
    engagement_variability[
        "engagement_variability"
    ]
    .fillna(0)
)

print("\nEngagement Variability Preview:")
print(engagement_variability.head())

# =========================
# INACTIVITY FEATURES
# =========================

print("\nCreating inactivity features...")

activity_range = (
    student_vle
    .groupby("id_student")["date"]
    .agg(["min", "max"])
    .reset_index()
)

activity_range.rename(
    columns={
        "min": "first_activity_day",
        "max": "last_activity_day"
    },
    inplace=True
)

activity_range = activity_range.merge(
    active_days,
    on="id_student",
    how="left"
)

activity_range["inactivity_days"] = (
    (
        activity_range["last_activity_day"]
        -
        activity_range["first_activity_day"]
    )
    -
    activity_range["active_days"]
)

activity_range[
    "inactivity_days"
] = (
    activity_range[
        "inactivity_days"
    ]
    .clip(lower=0)
)

print("\nInactivity Features Preview:")
print(activity_range.head())

# =========================
# ENGAGEMENT SLOPE
# =========================

print("\nCreating engagement slope features...")

daily_engagement = (
    student_vle
    .groupby(
        ["id_student", "date"]
    )["sum_click"]
    .sum()
    .reset_index()
)

def calculate_slope(group):

    if len(group) < 2:
        return 0

    x = group["date"]
    y = group["sum_click"]

    denominator = (
        (
            len(x)
            *
            (x ** 2).sum()
        )
        -
        (x.sum() ** 2)
    )

    if denominator == 0:
        return 0

    slope = (
        (
            (
                len(x)
                *
                (x * y).sum()
            )
            -
            (
                x.sum()
                *
                y.sum()
            )
        )
        /
        denominator
    )

    return slope

engagement_slope = (
    daily_engagement
    .groupby("id_student")
    .apply(calculate_slope)
    .reset_index(
        name="engagement_slope"
    )
)

print("\nEngagement Slope Preview:")
print(engagement_slope.head())

# =========================
# ASSESSMENT CONSISTENCY
# =========================

print(
    "\nCreating assessment consistency features..."
)

assessment_consistency = (
    student_assessment
    .groupby("id_student")["score"]
    .std()
    .reset_index()
)

assessment_consistency.rename(
    columns={
        "score":
        "assessment_consistency"
    },
    inplace=True
)

assessment_consistency[
    "assessment_consistency"
] = (
    assessment_consistency[
        "assessment_consistency"
    ]
    .fillna(0)
)

print("\nAssessment Consistency Preview:")
print(assessment_consistency.head())

# =========================
# MERGE ALL FEATURES
# =========================

merged_data = merged_data.merge(
    active_days,
    on="id_student",
    how="left"
)

merged_data = merged_data.merge(
    engagement_variability,
    on="id_student",
    how="left"
)

merged_data = merged_data.merge(
    activity_range[
        [
            "id_student",
            "inactivity_days"
        ]
    ],
    on="id_student",
    how="left"
)

merged_data = merged_data.merge(
    engagement_slope,
    on="id_student",
    how="left"
)

merged_data = merged_data.merge(
    assessment_consistency,
    on="id_student",
    how="left"
)

# =========================
# HANDLE REMAINING NULLS
# =========================

feature_columns = [
    "active_days",
    "engagement_variability",
    "inactivity_days",
    "engagement_slope",
    "assessment_consistency"
]

for col in feature_columns:

    merged_data[col] = (
        merged_data[col]
        .fillna(0)
    )

print("\nTemporal Features Added Successfully!")

print(
    merged_data[
        [
            "id_student",
            "active_days",
            "engagement_variability",
            "inactivity_days",
            "engagement_slope",
            "assessment_consistency"
        ]
    ].head()
)

# =========================
# MACHINE LEARNING FEATURES
# =========================

print("\nPreparing ML features...")

X = merged_data[
    [
        "total_clicks",
        "avg_score",
        "active_days",
        "engagement_variability",
        "inactivity_days",
        "engagement_slope",
        "assessment_consistency"
    ]
]

y = merged_data[
    "psychological_risk"
]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# =========================
# LOGISTIC REGRESSION
# =========================

print("\nTraining Logistic Regression model...")

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nLogistic Regression Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# =========================
# RANDOM FOREST
# =========================

print("\nTraining Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions
    )
)

print("\nRandom Forest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)

# =========================
# XGBOOST MODEL
# =========================

print("\nTraining XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = (
    xgb_model.predict(X_test)
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print("\nXGBoost Accuracy:")
print(xgb_accuracy)

print("\nXGBoost Classification Report:")
print(
    classification_report(
        y_test,
        xgb_predictions
    )
)

print("\nXGBoost Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        xgb_predictions
    )
)

# =========================
# SHAP EXPLAINABILITY
# =========================

print("\nGenerating SHAP explanations...")

explainer = shap.Explainer(
    xgb_model
)

shap_values = explainer(X_test)

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
    abs(
        shap_values.values
    ).mean(axis=0)

})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\nSHAP Feature Importance:")
print(feature_importance)

print(
    "\nSHAP explainability completed successfully!"
)

# =========================
# PERSONA CLUSTERING
# =========================

print(
    "\nPerforming student persona clustering..."
)

cluster_features = merged_data[
    [
        "total_clicks",
        "avg_score",
        "active_days",
        "engagement_variability",
        "inactivity_days",
        "engagement_slope",
        "assessment_consistency"
    ]
]

# =========================
# NORMALIZE FEATURES
# =========================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    cluster_features
)

# =========================
# IMPROVED KMEANS
# =========================

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=20,
    max_iter=500
)

merged_data["persona_cluster"] = (
    kmeans.fit_predict(
        scaled_features
    )
)

# =========================
# CLUSTER CENTER ANALYSIS
# =========================

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(
        kmeans.cluster_centers_
    ),
    columns=cluster_features.columns
)

print("\nCluster Centers:")
print(cluster_centers)

# =========================
# SMART PERSONA LABELING
# =========================

persona_labels = {}

for cluster_id in range(6):

    cluster = cluster_centers.iloc[
        cluster_id
    ]

    # Silent students
    if (
        cluster["inactivity_days"] > 180
        and cluster["active_days"] < 50
    ):

        persona_labels[
            cluster_id
        ] = "Silent Isolator"

    # Burnout students
    elif (
        cluster["engagement_slope"] < -0.05
    ):

        persona_labels[
            cluster_id
        ] = "Burnout Pattern"

    # Anxiety students
    elif (
        cluster[
            "engagement_variability"
        ] > 6
    ):

        persona_labels[
            cluster_id
        ] = (
            "Anxiety-Spike Learner"
        )

    # Consistent students
    elif (
        cluster["total_clicks"] > 2000
        and cluster["avg_score"] > 70
    ):

        persona_labels[
            cluster_id
        ] = "Consistent Learner"

    # Weak performers
    elif (
        cluster["avg_score"] < 40
    ):

        persona_labels[
            cluster_id
        ] = "Passive Watcher"

    # Default
    else:

        persona_labels[
            cluster_id
        ] = (
            "Last-Minute Survivor"
        )

# =========================
# ASSIGN PERSONAS
# =========================

merged_data["persona"] = (
    merged_data["persona_cluster"]
    .map(persona_labels)
)

print("\nPersona Distribution:")
print(
    merged_data["persona"]
    .value_counts()
)

print("\nPersona Preview:")

print(
    merged_data[
        [
            "id_student",
            "total_clicks",
            "avg_score",
            "active_days",
            "engagement_variability",
            "inactivity_days",
            "engagement_slope",
            "assessment_consistency",
            "persona"
        ]
    ].head(10)
)

print(
    "\nPersona clustering completed successfully!"
)



# =========================
# SAVE FINAL DATASET
# =========================

print("\nSaving final academic intelligence dataset...")

merged_data.to_csv(
    "../datasets/final_student_profiles.csv",
    index=False
)

print("\nFinal dataset saved successfully!")

# =========================
# SAVE XGBOOST MODEL
# =========================

import joblib

joblib.dump(
    xgb_model,
    "academic_risk_xgboost.pkl"
)

print(
    "XGBoost model saved successfully!"
)