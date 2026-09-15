
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)

BASE = "/content/drive/MyDrive/ecg_temporal_ai"

FEATURE_DIR = f"{BASE}/outputs/features"
FINAL_DIR = f"{BASE}/outputs/final_test_results"
EDA_DIR = f"{BASE}/outputs/eda"

os.makedirs(EDA_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

task_a = pd.read_csv(
    f"{FEATURE_DIR}/task_a_hrv_features.csv"
)

task_b = pd.read_csv(
    f"{FEATURE_DIR}/task_b_hrv_features.csv"
)

a_lr = pd.read_csv(
    f"{FINAL_DIR}/task_a_logistic_predictions.csv"
)

a_tcn = pd.read_csv(
    f"{FINAL_DIR}/task_a_tcn_predictions.csv"
)

b_lr = pd.read_csv(
    f"{FINAL_DIR}/task_b_logistic_predictions.csv"
)

b_tcn = pd.read_csv(
    f"{FINAL_DIR}/task_b_tcn_predictions.csv"
)


print("=" * 80)
print("STEP 11 — ERROR ANALYSIS AND CALIBRATION")
print("=" * 80)


# ============================================================
# HELPER: ADD ERROR CATEGORY
# ============================================================

def add_error_category(df):

    df = df.copy()

    df["error_type"] = np.select(
        [
            (df["target"] == 1) & (df["predicted_label"] == 1),
            (df["target"] == 0) & (df["predicted_label"] == 0),
            (df["target"] == 0) & (df["predicted_label"] == 1),
            (df["target"] == 1) & (df["predicted_label"] == 0),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="UNKNOWN",
    )

    return df


a_lr = add_error_category(a_lr)
a_tcn = add_error_category(a_tcn)
b_lr = add_error_category(b_lr)
b_tcn = add_error_category(b_tcn)


# ============================================================
# 1. SUBJECT-LEVEL ERROR ANALYSIS
# ============================================================

def subject_error_summary(
    df
):

    summary = (
        df.groupby("subject_id")
        .agg(
            n=("target", "size"),
            positives=("target", "sum"),
            false_positives=(
                "error_type",
                lambda x: (x == "FP").sum()
            ),
            false_negatives=(
                "error_type",
                lambda x: (x == "FN").sum()
            ),
            true_positives=(
                "error_type",
                lambda x: (x == "TP").sum()
            ),
            true_negatives=(
                "error_type",
                lambda x: (x == "TN").sum()
            ),
        )
        .reset_index()
    )

    summary["fp_rate"] = (
        summary["false_positives"]
        / np.maximum(
            summary["n"] - summary["positives"],
            1
        )
    )

    summary["fn_rate"] = (
        summary["false_negatives"]
        / np.maximum(
            summary["positives"],
            1
        )
    )

    return summary


for name, df in [
    ("task_a_logistic", a_lr),
    ("task_a_tcn", a_tcn),
    ("task_b_logistic", b_lr),
    ("task_b_tcn", b_tcn),
]:

    summary = subject_error_summary(df)

    summary.to_csv(
        f"{EDA_DIR}/{name}_subject_errors.csv",
        index=False
    )

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    display(
        summary.sort_values(
            "false_negatives",
            ascending=False
        ).head(10)
    )


# ============================================================
# 2. ERROR COUNTS
# ============================================================

error_summary = []

for name, df in [
    ("Task_A_Logistic", a_lr),
    ("Task_A_TCN", a_tcn),
    ("Task_B_Logistic", b_lr),
    ("Task_B_TCN", b_tcn),
]:

    counts = (
        df["error_type"]
        .value_counts()
        .to_dict()
    )

    error_summary.append({
        "model": name,
        "TP": counts.get("TP", 0),
        "TN": counts.get("TN", 0),
        "FP": counts.get("FP", 0),
        "FN": counts.get("FN", 0),
    })

error_summary = pd.DataFrame(
    error_summary
)

print("\n" + "=" * 80)
print("ERROR SUMMARY")
print("=" * 80)

display(error_summary)

error_summary.to_csv(
    f"{EDA_DIR}/error_summary.csv",
    index=False
)


# ============================================================
# 3. LINK ERRORS TO HRV QUALITY
# ============================================================

A_HRV_COLUMNS = [
    "subject_id",
    "minute_index",
    "mean_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
    "n_rr_valid",
    "rr_invalid_fraction",
]

a_quality = task_a[
    task_a["split"] == "test"
][A_HRV_COLUMNS].copy()


def merge_task_a_quality(
    predictions
):

    return predictions.merge(
        a_quality,
        on=[
            "subject_id",
            "minute_index"
        ],
        how="left"
    )


a_lr_q = merge_task_a_quality(a_lr)
a_tcn_q = merge_task_a_quality(a_tcn)


for name, df in [
    ("Task_A_Logistic", a_lr_q),
    ("Task_A_TCN", a_tcn_q),
]:

    error_quality = (
        df.groupby("error_type")[
            [
                "mean_hr_bpm",
                "sdnn_sec",
                "rmssd_sec",
                "pnn50",
                "cv_rr",
                "n_rr_valid",
                "rr_invalid_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )

    print("\n" + "-" * 80)
    print(name, "— mean feature values by error type")
    print("-" * 80)

    display(error_quality)

    error_quality.to_csv(
        f"{EDA_DIR}/{name.lower()}_error_quality.csv",
        index=False
    )


# ============================================================
# 4. TASK B ERROR QUALITY
# ============================================================

# Task B errors are analyzed using the 5-minute history.
# We summarize mean historical HR/HRV over the five input minutes.

B_FEATURES = [
    "mean_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
]

for name, df in [
    ("Task_B_Logistic", b_lr),
    ("Task_B_TCN", b_tcn),
]:

    enriched = df.copy()

    for feature in B_FEATURES:

        lag_columns = [
            f"{feature}_tminus{lag}"
            for lag in [5, 4, 3, 2, 1]
        ]

        enriched[
            f"{feature}_history_mean"
        ] = task_b.loc[
            task_b["split"] == "test",
            lag_columns
        ].mean(axis=1).values[
            :len(enriched)
        ]

    # Safer alignment using subject + target minute
    b_test_features = task_b[
        task_b["split"] == "test"
    ].copy()

    merge_columns = [
        "subject_id",
        "target_minute"
    ] + [
        f"{feature}_tminus{lag}"
        for feature in B_FEATURES
        for lag in [5, 4, 3, 2, 1]
    ]

    b_test_features = b_test_features[
        merge_columns
    ]

    enriched = enriched.merge(
        b_test_features,
        on=[
            "subject_id",
            "target_minute"
        ],
        how="left",
        suffixes=("", "_source")
    )

    history_summary = []

    for feature in B_FEATURES:

        cols = [
            f"{feature}_tminus{lag}"
            for lag in [5, 4, 3, 2, 1]
        ]

        enriched[
            f"{feature}_history_mean"
        ] = enriched[
            cols
        ].mean(axis=1)

    history_summary = (
        enriched.groupby("error_type")[
            [
                f"{feature}_history_mean"
                for feature in B_FEATURES
            ]
        ]
        .mean()
        .reset_index()
    )

    print("\n" + "-" * 80)
    print(name, "— history feature means by error type")
    print("-" * 80)

    display(history_summary)

    history_summary.to_csv(
        f"{EDA_DIR}/{name.lower()}_history_error_analysis.csv",
        index=False
    )


# ============================================================
# 5. TASK-B ERROR TIMING
# ============================================================

print("\n" + "=" * 80)
print("TASK B ERROR TIMING")
print("=" * 80)

# Examine the distance to the nearest previous/current apnea
# transition using the labels available in task_b.

b_context = task_b[
    task_b["split"] == "test"
][
    [
        "subject_id",
        "target_minute",
        "previous_label",
        "current_label",
        "target",
    ]
].copy()

b_lr_context = b_lr.merge(
    b_context,
    on=[
        "subject_id",
        "target_minute",
    ],
    how="left",
    suffixes=("", "_context")
)

b_tcn_context = b_tcn.merge(
    b_context,
    on=[
        "subject_id",
        "target_minute",
    ],
    how="left",
    suffixes=("", "_context")
)

for name, df in [
    ("Task_B_Logistic", b_lr_context),
    ("Task_B_TCN", b_tcn_context),
]:

    # "target=1 and previous_label=0" is the onset condition.
    # For errors, inspect whether predictions occur at actual onset
    # or non-onset minutes.

    timing = (
        df.groupby(
            [
                "error_type",
                "previous_label",
                "current_label",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    print("\n", name)
    display(timing)

    timing.to_csv(
        f"{EDA_DIR}/{name.lower()}_temporal_error_context.csv",
        index=False
    )


# ============================================================
# 6. CALIBRATION — ECE
# ============================================================

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10
):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    rows = []

    for i in range(n_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            mask = (
                (y_prob >= lower)
                & (y_prob <= upper)
            )
        else:
            mask = (
                (y_prob >= lower)
                & (y_prob < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_probability = float(
            y_prob[mask].mean()
        )

        empirical_frequency = float(
            y_true[mask].mean()
        )

        gap = abs(
            mean_probability
            - empirical_frequency
        )

        weight = (
            count / len(y_true)
        )

        ece += (
            weight * gap
        )

        rows.append({
            "bin": i,
            "count": count,
            "mean_probability": mean_probability,
            "empirical_frequency": empirical_frequency,
            "absolute_gap": gap,
        })

    return (
        float(ece),
        pd.DataFrame(rows)
    )


calibration_results = []

for name, df in [
    ("Task_A_Logistic", a_lr),
    ("Task_A_TCN", a_tcn),
    ("Task_B_Logistic", b_lr),
    ("Task_B_TCN", b_tcn),
]:

    y_true = df["target"].values
    y_prob = df[
        "predicted_probability"
    ].values

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    ece, calibration_table = (
        expected_calibration_error(
            y_true,
            y_prob,
            n_bins=10
        )
    )

    calibration_results.append({
        "model": name,
        "brier": brier,
        "ece": ece,
    })

    calibration_table.to_csv(
        f"{EDA_DIR}/{name.lower()}_reliability_bins.csv",
        index=False
    )

calibration_results = pd.DataFrame(
    calibration_results
)

print("\n" + "=" * 80)
print("CALIBRATION RESULTS")
print("=" * 80)

display(
    calibration_results
)

calibration_results.to_csv(
    f"{EDA_DIR}/calibration_results.csv",
    index=False
)


# ============================================================
# 7. RELIABILITY PLOTS
# ============================================================

for name, df in [
    ("Task A Logistic", a_lr),
    ("Task A TCN", a_tcn),
    ("Task B Logistic", b_lr),
    ("Task B TCN", b_tcn),
]:

    ece, reliability = (
        expected_calibration_error(
            df["target"].values,
            df["predicted_probability"].values,
            n_bins=10
        )
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        reliability["mean_probability"],
        reliability["empirical_frequency"],
        marker="o",
        label="Model"
    )

    plt.xlabel(
        "Predicted probability"
    )

    plt.ylabel(
        "Observed frequency"
    )

    plt.title(
        f"Reliability diagram — {name}"
    )

    plt.legend()

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        f"{EDA_DIR}/{safe_name}_reliability.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 COMPLETE")
print("=" * 80)

print(
    "\nSaved error-analysis and calibration outputs to:"
)

print(
    EDA_DIR
)

print("\nCalibration:")
display(calibration_results)
