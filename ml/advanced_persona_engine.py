import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Loading academic intelligence dataset...")

# =========================
# LOAD ACADEMIC DATA
# =========================

academic_df = pd.read_csv(
    "../datasets/final_student_profiles.csv"
)

print("\nAcademic Dataset Loaded!")

print("\nAcademic Dataset Shape:")
print(academic_df.shape)

# =========================
# LOAD COGNITIVE DATA
# =========================

print("\nLoading cognitive intelligence dataset...")

cognitive_df = pd.read_csv(
    "video_intelligence/cognitive_behavior_results.csv"
)

print("\nCognitive Dataset Loaded!")

print("\nCognitive Dataset Shape:")
print(cognitive_df.shape)

# =========================
# PREPARE COGNITIVE FEATURES
# =========================

cognitive_features = cognitive_df[
    [
        "attention_score",
        "confusion_score",
        "boredom_score",
        "cognitive_overload_score"
    ]
].copy()

# =========================
# SYNTHETIC MAPPING
# =========================

print(
    "\nMapping cognitive intelligence to students..."
)

repeat_count = int(
    np.ceil(
        len(academic_df)
        / len(cognitive_features)
    )
)

expanded_cognitive = pd.concat(
    [cognitive_features] * repeat_count,
    ignore_index=True
)

expanded_cognitive = expanded_cognitive.iloc[
    :len(academic_df)
]

expanded_cognitive.reset_index(
    drop=True,
    inplace=True
)

# =========================
# MERGE DATA
# =========================

persona_df = pd.concat(
    [
        academic_df.reset_index(drop=True),
        expanded_cognitive
    ],
    axis=1
)

print(
    "\nMerged Multimodal Dataset Shape:"
)

print(persona_df.shape)

# =========================
# SELECT FEATURES
# =========================

print(
    "\nPreparing persona intelligence features..."
)

persona_features = persona_df[
    [
        "total_clicks",
        "avg_score",
        "active_days",
        "engagement_variability",
        "inactivity_days",
        "engagement_slope",
        "assessment_consistency",
        "attention_score",
        "confusion_score",
        "boredom_score",
        "cognitive_overload_score"
    ]
]

# =========================
# SCALE FEATURES
# =========================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    persona_features
)

# =========================
# KMEANS CLUSTERING
# =========================

print(
    "\nTraining multimodal persona clustering engine..."
)

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

persona_clusters = kmeans.fit_predict(
    scaled_features
)

persona_df["persona_cluster"] = (
    persona_clusters
)

# =========================
# PERSONA LABELS
# =========================

persona_map = {

    0: "Silent Isolator",

    1: "Burnout Pattern",

    2: "Anxiety-Spike Learner",

    3: "Passive Watcher",

    4: "Consistent Learner",

    5: "Last-Minute Survivor"

}

persona_df["persona"] = persona_df[
    "persona_cluster"
].map(
    persona_map
)

# =========================
# STABILITY SCORE
# =========================

print(
    "\nGenerating behavioral stability scores..."
)

persona_df["stability_score"] = (

    100
    - (

        persona_df[
            "engagement_variability"
        ]

        + persona_df[
            "confusion_score"
        ]

        + persona_df[
            "cognitive_overload_score"
        ]

    ) / 3

)

persona_df["stability_score"] = (
    persona_df["stability_score"]
    .clip(
        lower=0,
        upper=100
    )
)

# =========================
# INTERVENTION STYLE
# =========================

def intervention_strategy(
    persona
):

    if persona == "Silent Isolator":

        return (
            "Social engagement support"
        )

    elif persona == "Burnout Pattern":

        return (
            "Mental wellness intervention"
        )

    elif persona == (
        "Anxiety-Spike Learner"
    ):

        return (
            "Guided pacing and stress reduction"
        )

    elif persona == (
        "Passive Watcher"
    ):

        return (
            "Interactive participation encouragement"
        )

    elif persona == (
        "Consistent Learner"
    ):

        return (
            "Advanced learning opportunities"
        )

    else:

        return (
            "Deadline management coaching"
        )

persona_df[
    "intervention_style"
] = (
    persona_df["persona"]
    .apply(
        intervention_strategy
    )
)

# =========================
# PERSONA DISTRIBUTION
# =========================

print(
    "\nPersona Distribution:"
)

print(
    persona_df["persona"]
    .value_counts()
)

# =========================
# PERSONA PREVIEW
# =========================

print(
    "\nPersona Intelligence Preview:"
)

print(

    persona_df[
        [
            "id_student",
            "persona",
            "stability_score",
            "attention_score",
            "confusion_score",
            "cognitive_overload_score",
            "intervention_style"
        ]
    ].head(10)

)

# =========================
# CLUSTER CENTERS
# =========================

cluster_centers = pd.DataFrame(

    kmeans.cluster_centers_,

    columns=
    persona_features.columns

)

print(
    "\nCluster Centers:"
)

print(
    cluster_centers
)

# =========================
# SAVE DATASET
# =========================

persona_df.to_csv(
    "advanced_persona_profiles.csv",
    index=False
)

print(
    "\nAdvanced persona profiles saved successfully!"
)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    kmeans,
    "persona_kmeans.pkl"
)

joblib.dump(
    scaler,
    "persona_scaler.pkl"
)

print(
    "\nPersona model saved successfully!"
)

print(
    "\nAdvanced Multimodal Persona Intelligence Completed!"
)