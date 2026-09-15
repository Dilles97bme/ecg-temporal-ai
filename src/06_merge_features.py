
import os
import pandas as pd
import numpy as np
from IPython.display import display
# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/content/drive/MyDrive/ecg_temporal_ai"

WINDOW_DIR = f"{BASE}/outputs/windowing"
FEATURE_DIR = f"{BASE}/outputs/features"

TASK_A_PATH = f"{WINDOW_DIR}/task_a_windows.csv"
TASK_B_PATH = f"{WINDOW_DIR}/task_b_windows.csv"
HRV_PATH = f"{FEATURE_DIR}/all_records_hrv_features.csv"

TASK_A_OUT = f"{FEATURE_DIR}/task_a_hrv_features.csv"
TASK_B_OUT = f"{FEATURE_DIR}/task_b_hrv_features.csv"
QC_OUT = f"{FEATURE_DIR}/feature_merge_qc.csv"

os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

task_a = pd.read_csv(TASK_A_PATH)
task_b = pd.read_csv(TASK_B_PATH)
hrv = pd.read_csv(HRV_PATH)

print("=" * 80)
print("STEP 6: FINAL HRV FEATURE MERGE")
print("=" * 80)

print("\nInput shapes:")
print("Task A:", task_a.shape)
print("Task B:", task_b.shape)
print("HRV   :", hrv.shape)


# ============================================================
# HRV FEATURES
# ============================================================

HRV_FEATURES = [
    "mean_rr_sec",
    "median_rr_sec",
    "mean_hr_bpm",
    "median_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
    "n_rr_total",
    "n_rr_valid",
    "rr_invalid_fraction",
    "feature_valid",
]


# ============================================================
# HRV KEY QC
# ============================================================

duplicates = hrv.duplicated(
    ["subject_id", "minute"]
).sum()

print("\nDuplicate HRV subject-minute keys:", duplicates)

assert duplicates == 0


# ============================================================
# TASK A
# ============================================================

print("\n" + "-" * 80)
print("TASK A — CURRENT APNEA DETECTION")
print("-" * 80)

hrv_a = hrv.rename(
    columns={"minute": "minute_index"}
)

task_a_merged = task_a.merge(
    hrv_a[
        ["subject_id", "minute_index"] + HRV_FEATURES
    ],
    on=["subject_id", "minute_index"],
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# HRV availability
# ------------------------------------------------------------

task_a_merged["hrv_available"] = (
    task_a_merged["feature_valid"].notna()
)

task_a_merged["hrv_complete_valid"] = (
    task_a_merged["feature_valid"] == 1
)

print("\nTask A HRV availability:")
print(
    task_a_merged["hrv_available"]
    .value_counts(dropna=False)
)

print("\nTask A HRV validity:")
print(
    task_a_merged["hrv_complete_valid"]
    .value_counts(dropna=False)
)

print("\nTask A validity by split:")
display(
    task_a_merged
    .groupby("split")["hrv_complete_valid"]
    .value_counts()
    .unstack(fill_value=0)
)

print("\nTask A validity by target:")
display(
    task_a_merged
    .groupby("target")["hrv_complete_valid"]
    .value_counts()
    .unstack(fill_value=0)
)


# ============================================================
# TASK B
# ============================================================

print("\n" + "-" * 80)
print("TASK B — 5-MINUTE HISTORY -> NEXT-MINUTE ONSET")
print("-" * 80)

hrv_lookup = hrv.set_index(
    ["subject_id", "minute"]
)

task_b_rows = []

for _, row in task_b.iterrows():

    subject = row["subject_id"]

    target_minute = int(
        row["target_minute"]
    )

    history_start = int(
        row["history_start_minute"]
    )

    history_end = int(
        row["history_end_minute"]
    )

    history_minutes = list(
        range(history_start, history_end)
    )

    # --------------------------------------------------------
    # Temporal definition checks
    # --------------------------------------------------------

    assert len(history_minutes) == 5

    assert target_minute not in history_minutes

    assert history_end == target_minute

    output = row.to_dict()

    valid_flags = []

    # --------------------------------------------------------
    # Add lagged HRV features
    # --------------------------------------------------------

    for lag, minute in zip(
        [5, 4, 3, 2, 1],
        history_minutes
    ):

        key = (subject, minute)

        if key in hrv_lookup.index:

            hrow = hrv_lookup.loc[key]

            for feature in HRV_FEATURES:

                output[
                    f"{feature}_tminus{lag}"
                ] = hrow[feature]

            valid_flags.append(
                hrow["feature_valid"] == 1
            )

        else:

            for feature in HRV_FEATURES:

                output[
                    f"{feature}_tminus{lag}"
                ] = np.nan

            valid_flags.append(False)

    # --------------------------------------------------------
    # History-level quality
    # --------------------------------------------------------

    output["history_valid_count"] = int(
        sum(valid_flags)
    )

    output["history_complete_valid"] = int(
        all(valid_flags)
    )

    task_b_rows.append(output)


task_b_merged = pd.DataFrame(task_b_rows)


# ============================================================
# TASK B FEATURE LIST
# ============================================================

TASK_B_FEATURES = []

for lag in [5, 4, 3, 2, 1]:

    for feature in HRV_FEATURES:

        TASK_B_FEATURES.append(
            f"{feature}_tminus{lag}"
        )


# ============================================================
# TASK B QC
# ============================================================

print("\nTask B merged shape:")
print(task_b_merged.shape)

print("\nHistory valid-count distribution:")
print(
    task_b_merged[
        "history_valid_count"
    ].value_counts()
    .sort_index()
)

print("\nComplete valid histories:")
print(
    task_b_merged[
        "history_complete_valid"
    ].value_counts()
)


print("\nTask B complete-history status by split:")
display(
    task_b_merged
    .groupby("split")[
        "history_complete_valid"
    ]
    .value_counts()
    .unstack(fill_value=0)
)


print("\nTask B complete-history status by target:")
display(
    task_b_merged
    .groupby("target")[
        "history_complete_valid"
    ]
    .value_counts()
    .unstack(fill_value=0)
)


# ============================================================
# CRITICAL LEAKAGE CHECK
# ============================================================

print("\n" + "-" * 80)
print("TEMPORAL LEAKAGE CHECK")
print("-" * 80)

assert (
    task_b_merged["history_end_minute"]
    == task_b_merged["target_minute"]
).all()

assert (
    task_b_merged["history_end_minute"]
    - task_b_merged["history_start_minute"]
    == 5
).all()

print(
    "PASS: Every Task B history contains exactly five minutes."
)

print(
    "PASS: History ends immediately before target minute."
)

print(
    "PASS: Target minute HRV is NOT included as input."
)


# ============================================================
# POSITIVE-EVENT CHECK
# ============================================================

b_positive = task_b_merged[
    task_b_merged["target"] == 1
]

b_positive_complete = b_positive[
    b_positive["history_complete_valid"] == 1
]

print("\nTask B positive events:", len(b_positive))
print(
    "Positive events with complete valid history:",
    len(b_positive_complete)
)

assert len(b_positive) == len(
    b_positive_complete
)

print(
    "PASS: All positive Task B events have "
    "complete valid 5-minute HRV histories."
)


# ============================================================
# PRIMARY ANALYSIS FLAGS
# ============================================================

# Task A primary HRV analysis
task_a_merged["primary_hrv_analysis"] = (
    task_a_merged["hrv_complete_valid"]
    .astype(int)
)

# Task B primary HRV analysis
task_b_merged["primary_hrv_analysis"] = (
    task_b_merged["history_complete_valid"]
    .astype(int)
)


# ============================================================
# SAVE
# ============================================================

task_a_merged.to_csv(
    TASK_A_OUT,
    index=False
)

task_b_merged.to_csv(
    TASK_B_OUT,
    index=False
)


# ============================================================
# QC SUMMARY
# ============================================================

qc = pd.DataFrame({

    "dataset": [
        "Task_A",
        "Task_B"
    ],

    "total_rows": [
        len(task_a_merged),
        len(task_b_merged)
    ],

    "primary_valid_rows": [
        int(
            task_a_merged[
                "primary_hrv_analysis"
            ].sum()
        ),
        int(
            task_b_merged[
                "primary_hrv_analysis"
            ].sum()
        )
    ],

    "excluded_invalid_rows": [
        int(
            (~task_a_merged[
                "primary_hrv_analysis"
            ].astype(bool)).sum()
        ),
        int(
            (~task_b_merged[
                "primary_hrv_analysis"
            ].astype(bool)).sum()
        )
    ],

    "positive_rows": [
        int(task_a_merged["target"].sum()),
        int(task_b_merged["target"].sum())
    ],

    "positive_rows_excluded": [
        int(
            task_a_merged.loc[
                ~task_a_merged[
                    "primary_hrv_analysis"
                ].astype(bool),
                "target"
            ].sum()
        ),
        int(
            task_b_merged.loc[
                ~task_b_merged[
                    "primary_hrv_analysis"
                ].astype(bool),
                "target"
            ].sum()
        )
    ]
})

qc.to_csv(
    QC_OUT,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 COMPLETE")
print("=" * 80)

print("\nTask A final:")
print("  Total:", len(task_a_merged))
print(
    "  Primary valid:",
    task_a_merged[
        "primary_hrv_analysis"
    ].sum()
)
print(
    "  Excluded:",
    (~task_a_merged[
        "primary_hrv_analysis"
    ].astype(bool)).sum()
)

print("\nTask B final:")
print("  Total:", len(task_b_merged))
print(
    "  Primary valid:",
    task_b_merged[
        "primary_hrv_analysis"
    ].sum()
)
print(
    "  Excluded:",
    (~task_b_merged[
        "primary_hrv_analysis"
    ].astype(bool)).sum()
)

print("\nSaved:")
print(TASK_A_OUT)
print(TASK_B_OUT)
print(QC_OUT)
