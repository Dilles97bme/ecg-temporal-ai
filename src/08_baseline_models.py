
import os
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
)

from IPython.display import display

warnings.filterwarnings("ignore")


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/content/drive/MyDrive/ecg_temporal_ai"

FEATURE_DIR = f"{BASE}/outputs/features"
RESULT_DIR = f"{BASE}/outputs/baseline_results"

os.makedirs(RESULT_DIR, exist_ok=True)

TASK_A_PATH = f"{FEATURE_DIR}/task_a_hrv_features.csv"
TASK_B_PATH = f"{FEATURE_DIR}/task_b_hrv_features.csv"

RESULT_PATH = f"{RESULT_DIR}/baseline_validation_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

task_a = pd.read_csv(TASK_A_PATH)
task_b = pd.read_csv(TASK_B_PATH)

print("=" * 80)
print("STEP 8 — NON-DEEP BASELINE MODELING")
print("=" * 80)

print("\nTask A:", task_a.shape)
print("Task B:", task_b.shape)

print("\nIMPORTANT:")
print("Test set is NOT used in this script.")
print("Only training and validation are used for model development.")


# ============================================================
# COMMON METRICS
# ============================================================

def evaluate_validation(
    y_true,
    y_prob,
    threshold,
    task_name,
    model_name,
):
    """
    Evaluate predictions on the VALIDATION set only.
    Test evaluation is intentionally excluded from this script.
    """

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    auroc = roc_auc_score(
        y_true,
        y_prob
    )

    auprc = average_precision_score(
        y_true,
        y_prob
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    prevalence = float(
        np.mean(y_true)
    )

    return {
        "task": task_name,
        "model": model_name,
        "validation_positive_prevalence": prevalence,
        "auprc": auprc,
        "auroc": auroc,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": f1,
        "accuracy": accuracy,
        "brier": brier,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# VALIDATION THRESHOLD SELECTION
# ============================================================

def select_f1_threshold(
    y_true,
    y_prob
):
    """
    Select threshold ONLY using validation labels.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_idx = int(
        np.argmax(f1_values)
    )

    return float(
        thresholds[best_idx]
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def make_models():

    return {

        "LogisticRegression": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED
                )
            )
        ]),

        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                    max_features="sqrt"
                )
            )
        ])
    }


# ============================================================
# HRV FEATURE DEFINITIONS
# ============================================================

BASE_HRV_FEATURES = [
    "mean_rr_sec",
    "median_rr_sec",
    "mean_hr_bpm",
    "median_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
]


# ============================================================
# TASK A
# ============================================================

print("\n" + "=" * 80)
print("TASK A — CURRENT APNEA DETECTION")
print("=" * 80)

a = task_a[
    task_a["primary_hrv_analysis"] == 1
].copy()

print(
    "Primary valid Task A rows:",
    len(a)
)

print(
    "Subjects:",
    a["subject_id"].nunique()
)

train_mask = (
    a["split"] == "train"
)

val_mask = (
    a["split"] == "validation"
)

X_train_a = a.loc[
    train_mask,
    BASE_HRV_FEATURES
]

y_train_a = a.loc[
    train_mask,
    "target"
].astype(int)

X_val_a = a.loc[
    val_mask,
    BASE_HRV_FEATURES
]

y_val_a = a.loc[
    val_mask,
    "target"
].astype(int)

print(
    "\nTraining:",
    len(y_train_a),
    "positive:",
    int(y_train_a.sum())
)

print(
    "Validation:",
    len(y_val_a),
    "positive:",
    int(y_val_a.sum())
)

print(
    "Test rows:",
    int((a["split"] == "test").sum()),
    "(NOT USED)"
)


# ============================================================
# TASK A BASELINES
# ============================================================

results = []

for model_name, model in make_models().items():

    print(
        f"\nTraining Task A — {model_name}"
    )

    model.fit(
        X_train_a,
        y_train_a
    )

    val_prob = model.predict_proba(
        X_val_a
    )[:, 1]

    threshold = select_f1_threshold(
        y_val_a,
        val_prob
    )

    metrics = evaluate_validation(
        y_true=y_val_a,
        y_prob=val_prob,
        threshold=threshold,
        task_name="Task_A",
        model_name=model_name
    )

    results.append(metrics)

    print(
        "Validation AUROC:",
        round(metrics["auroc"], 4)
    )

    print(
        "Validation AUPRC:",
        round(metrics["auprc"], 4)
    )

    print(
        "Validation prevalence:",
        round(
            metrics[
                "validation_positive_prevalence"
            ],
            4
        )
    )

    print(
        "No-skill AUPRC baseline:",
        round(
            metrics[
                "validation_positive_prevalence"
            ],
            4
        )
    )

    print(
        "Validation threshold:",
        round(
            metrics["threshold"],
            4
        )
    )

    print(
        "Validation F1:",
        round(
            metrics["f1"],
            4
        )
    )


# ============================================================
# TASK B FEATURE DEFINITIONS
# ============================================================

TASK_B_FEATURES = []

for lag in [5, 4, 3, 2, 1]:

    for feature in BASE_HRV_FEATURES:

        TASK_B_FEATURES.append(
            f"{feature}_tminus{lag}"
        )


# ============================================================
# TASK B
# ============================================================

print("\n" + "=" * 80)
print("TASK B — 5-MINUTE HISTORY -> NEXT-MINUTE ONSET")
print("=" * 80)

b = task_b[
    task_b["primary_hrv_analysis"] == 1
].copy()

print(
    "Primary valid Task B rows:",
    len(b)
)

print(
    "Subjects:",
    b["subject_id"].nunique()
)

train_mask = (
    b["split"] == "train"
)

val_mask = (
    b["split"] == "validation"
)

X_train_b = b.loc[
    train_mask,
    TASK_B_FEATURES
]

y_train_b = b.loc[
    train_mask,
    "target"
].astype(int)

X_val_b = b.loc[
    val_mask,
    TASK_B_FEATURES
]

y_val_b = b.loc[
    val_mask,
    "target"
].astype(int)

print(
    "\nTraining:",
    len(y_train_b),
    "positive:",
    int(y_train_b.sum())
)

print(
    "Validation:",
    len(y_val_b),
    "positive:",
    int(y_val_b.sum())
)

print(
    "Test rows:",
    int((b["split"] == "test").sum()),
    "(NOT USED)"
)


# ============================================================
# TASK B BASELINES
# ============================================================

for model_name, model in make_models().items():

    print(
        f"\nTraining Task B — {model_name}"
    )

    model.fit(
        X_train_b,
        y_train_b
    )

    val_prob = model.predict_proba(
        X_val_b
    )[:, 1]

    threshold = select_f1_threshold(
        y_val_b,
        val_prob
    )

    metrics = evaluate_validation(
        y_true=y_val_b,
        y_prob=val_prob,
        threshold=threshold,
        task_name="Task_B",
        model_name=model_name
    )

    results.append(metrics)

    print(
        "Validation AUROC:",
        round(metrics["auroc"], 4)
    )

    print(
        "Validation AUPRC:",
        round(metrics["auprc"], 4)
    )

    print(
        "Validation prevalence:",
        round(
            metrics[
                "validation_positive_prevalence"
            ],
            4
        )
    )

    print(
        "No-skill AUPRC baseline:",
        round(
            metrics[
                "validation_positive_prevalence"
            ],
            4
        )
    )

    print(
        "Validation threshold:",
        round(
            metrics["threshold"],
            6
        )
    )

    print(
        "Validation F1:",
        round(
            metrics["f1"],
            4
        )
    )


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\n" + "=" * 80)
print("VALIDATION BASELINE RESULTS")
print("=" * 80)

display(results_df)

print(
    "\nSaved:",
    RESULT_PATH
)

print(
    "\nTEST SET WAS NOT USED FOR MODEL SELECTION "
    "OR THRESHOLD SELECTION."
)
